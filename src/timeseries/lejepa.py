# changeset: improved TemporalSIGReg + JEPA training loop
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchvision.ops import MLP
from .models.covariate import FutureCovariateEncoder


class TemporalSIGReg(nn.Module):

    """
    Vectorized SIGReg adapted for temporal embeddings.

    - input z: [N_samples, D]  where N_samples = B * T (or number of embeddings you want tested)
    - the test projects onto `num_slices` random directions (1D slices) and applies a univariate test per slice.
    - returns scalar loss (mean across slices).
    """
    def __init__(self, feature_dim, num_slices=1024, knots=17, A_dim=256, seed=0):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_slices = num_slices
        self.knots = knots
        self.A_dim = A_dim
        self.seed = seed

        # create the univariate test weights (same as LeJEPA: t grid, window, weights)
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)                 # [K]
        self.register_buffer("phi", window)          # [K]
        self.register_buffer("weights", weights * window)  # [K]

        # NOTE: We intentionally do NOT store a fixed projection matrix.
        # We re-sample slices each forward pass (seeded by global_step) to better match SIGReg.

    def _sample_A(self, *, device: torch.device, dtype: torch.dtype, global_step: int | None):
        rng = torch.Generator(device=device)
        if global_step is None:
            rng.manual_seed(self.seed)
        else:
            # Deterministic across DDP ranks if everyone uses the same global_step.
            rng.manual_seed(int(self.seed) + int(global_step))

        A = torch.randn(self.feature_dim, self.num_slices, generator=rng, device=device, dtype=dtype)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-12)
        return A

    def forward(self, z, global_step: int | None = None):
        """
        z: [B, T, D] OR [N, D]; we accept both
        returns scalar loss
        """
        if z.dim() == 3:
            B, T, D = z.shape
            z_flat = z.reshape(-1, D)   # [B*T, D]
        else:
            z_flat = z
        # z_flat: [N, D]
        # project: [N, S] = z_flat @ A
        A = self._sample_A(device=z_flat.device, dtype=z_flat.dtype, global_step=global_step)
        proj = z_flat @ A        # [N, S]

        # we want to evaluate characteristic function at several t values (knots)
        # expand proj to [N, S, K] by outer product with self.t
        # proj.unsqueeze(-1): [N, S, 1], self.t: [K] -> broadcast to [N, S, K]
        x_t = proj.unsqueeze(-1) * self.t.view(1, 1, -1)  # [N, S, K]

        # compute means over N (samples): mean over axis 0 -> [S, K]
        cos_mean = x_t.cos().mean(dim=0)  # [S, K]
        sin_mean = x_t.sin().mean(dim=0)  # [S, K]

        # err per slice & knot: [S, K]
        err = (cos_mean - self.phi.view(1, -1)).square() + sin_mean.square()

        # integrate over knots with weights -> [S]
        per_slice = err @ self.weights  # [S]

        # mean across slices
        loss = per_slice.mean()
        return loss


class LeJEPA_Forecaster(L.LightningModule):

    def __init__(self, encoder_backbone, input_dim, horizon,
                 proj_dim=128, num_slices=1024, lamb=0.5, lr=1e-3,
                 scaler_mean=None, scaler_std=None,
                 use_covariates=False, num_time_features=6,
                 reg_type="sigreg", vicreg_gamma=1.0, vicreg_mu=1.0):
        """
        Args:
            reg_type: "sigreg" (default) or "vicreg" for regularization type
            vicreg_gamma: VICReg variance target std (only used if reg_type="vicreg")
            vicreg_mu: VICReg covariance loss weight (only used if reg_type="vicreg")
        """
        super().__init__()
        self.save_hyperparameters(ignore=['encoder_backbone'])
        self.backbone = encoder_backbone                 # encoder should accept windows and return [B, T, D] or [B, D]
        self.backbone_out = getattr(encoder_backbone, "output_dim", None)
        if self.backbone_out is None:
            raise ValueError("encoder_backbone must expose .output_dim")
        
        # Covariate handling
        self.use_covariates = use_covariates
        self.num_time_features = num_time_features

        # Projector: map backbone outputs to proj_dim
        # We use a 2-layer MLP (Blue configuration) but switch to BatchNorm1d for better SSL statistics
        # and remove Dropout.
        self.proj = nn.Sequential(
            nn.Linear(self.backbone_out, proj_dim*2),
            nn.BatchNorm1d(proj_dim*2),
            nn.GELU(),
            nn.Linear(proj_dim*2, proj_dim),
        )

        # Future covariate encoder (for known future time features)
        if use_covariates:
            self.future_cov_encoder = FutureCovariateEncoder(
                input_features=num_time_features, 
                hidden_dim=32, 
                output_dim=64
            )
            forecast_input_dim = self.backbone_out + self.future_cov_encoder.output_dim
        else:
            self.future_cov_encoder = None
            forecast_input_dim = self.backbone_out

        # Predictor: map context projected embeddings -> target projected embeddings
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),  # maintained dim
            nn.LayerNorm(proj_dim),         # added norm for stability
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)   # output delta
        )

        # Regularizer: SIGReg or VICReg
        self.reg_type = reg_type
        if reg_type == "vicreg":
            self.regularizer = TemporalVICReg(feature_dim=proj_dim, gamma=vicreg_gamma, mu=vicreg_mu)
        else:
            self.regularizer = TemporalSIGReg(feature_dim=proj_dim, num_slices=num_slices, A_dim=256)

        # Forecasting head
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(forecast_input_dim),
            nn.Linear(forecast_input_dim, input_dim*horizon)
        )
        self.horizon = horizon
        self.input_dim = input_dim
        self.lr = lr
        self.lamb = lamb

        # Scaler parameters for unscaled metrics
        if scaler_mean is not None and scaler_std is not None:
            # Expect [1, C, 1] shape from TorchStandardScaler
            self.register_buffer("scaler_mean", scaler_mean)
            self.register_buffer("scaler_std", scaler_std)
        else:
            self.scaler_mean = None
            self.scaler_std = None
        torch.set_float32_matmul_precision('high') # we lets it be here to later use with ray?

        
    def forward(self, x, time_features=None):
        """Standard forward pass: encode a single window or multiple views."""
        # If x is [B, C, L], just encode it
        # If x is [B, V, C, L], we typically want to encode the current view (V=1) or mean of views
        if x.ndim == 4:
            B, V, C, L = x.shape
            x = x.view(B * V, C, L)
            
            if time_features is not None:
                # Flatten time_features to [B*V, ...]
                tf_flat = time_features.reshape(B * V, *time_features.shape[2:])
                emb = self._backbone_forward(x, tf_flat)
            else:
                emb = self._backbone_forward(x)

            # NOTE: This reshape assumes the backbone returns [B*V, D]. If the backbone returns
            # per-time embeddings [B*V, T, D], this will silently flatten (T*D) and lose structure.
            # If you ever use this path with a token-level backbone, revisit this return shape.
            return emb.view(B, V, -1)
            
        return self._backbone_forward(x, time_features)

    def _backbone_forward(self, x, time_features=None):
        """Call backbone with optional time_features.

        Some backbones (e.g., simple CNN encoders) only accept the window tensor.
        In that case we silently ignore time_features.
        """
        if time_features is None:
            return self.backbone(x)
        try:
            return self.backbone(x, time_features)
        except TypeError:
            return self.backbone(x)

    def encode_window(self, window, time_features=None):
        """
        window: [B, C, T] or [B, T, C] depending on your backbone
        time_features: Optional temporal covariates
        returns: embeddings per token/time-step [B, T, D] OR pooled [B, D]
        """
        return self._backbone_forward(window, time_features)

    def shared_step(self, batch, batch_idx, mode="train"):
        """
        Expect dataset to return:
                    views: [B, V, C, L] with V >= 3
          y_true: [B, H, C] (actual future numeric targets, transposed)
          view_times: [B, V, L, 6] - time features for each view (optional)
          future_times: [B, H, 6] - time features for forecast horizon (optional)
        """
        # Safety guard: cuDNN RNNs (LSTM/GRU) require training-mode forward
        # if we will backprop through them. If something (e.g., a notebook cell,
        # callback, or probe code) accidentally called `.eval()` on the backbone,
        # training would crash with: "cudnn RNN backward can only be called in training mode".
        if mode == "train" and hasattr(self, "backbone") and not self.backbone.training:
            self.backbone.train(True)

        # Unpack batch - handle both old (2-tuple) and new (4-tuple) format
        if len(batch) == 4:
            views, y_true, view_times, future_times = batch
        else:
            views, y_true = batch
            view_times, future_times = None, None
        
        B, V, C, L = views.shape
        if V < 3:
            raise ValueError(
                f"Expected at least 3 views (prev, curr, next). Got V={V}. "
                "Check dataset repeat_factor and view construction."
            )

        # 1) Encode all views
        # views: [B, V, C, L] -> flatten batch and view dimensions for encoder
        views_flat = views.view(B * V, C, L)
        
        if view_times is not None:
            # Flatten view_times [B, V, L, 6] -> [B*V, L, 6]
            view_times_flat = view_times.reshape(B * V, *view_times.shape[2:])
            emb_all = self.encode_window(views_flat, view_times_flat)
        else:
            emb_all = self.encode_window(views_flat)  # [B*V, D] or [B*V, T, D]

        def project(emb):
            if emb.dim() == 3:
                BV, T, D = emb.shape
                emb_reshape = emb.reshape(-1, D)
                z_flat = self.proj(emb_reshape)
                return z_flat.view(BV, T, -1)
            else:
                return self.proj(emb)

        z_all = project(emb_all)  # [B*V, P] or [B*V, T, P]
        
        # Pool if needed
        if z_all.dim() == 3:
            pooled_all = z_all.mean(dim=1)  # [B*V, P]
        else:
            pooled_all = z_all  # [B*V, P]
            
        # Reshape back to [B, V, P]
        pooled_views = pooled_all.view(B, V, -1)
        # View convention:
        # - index 0: previous window (t-1)
        # - indices 1..V-2: repeated augmented current window(s) (t0)
        # - index V-1: next window (t+1)
        z_prev = pooled_views[:, 0, :]
        z_next = pooled_views[:, -1, :]
        z_curr_repeats = pooled_views[:, 1:-1, :]  # [B, R, P]
        z_curr = z_curr_repeats.mean(dim=1)        # [B, P]
        # 2) JEPA Loss: 
        # a) Consistency loss: ONLY on augmentations of the same window (t0)
        # pooled_views shape [B, Views, P]. We need variance across the repeated views of t0.
        
        curr_mean = z_curr_repeats.mean(dim=1, keepdim=True) # [B, 1, P]
        temp_inv_loss = (z_curr_repeats - curr_mean).square().mean()
       
        
        
        # Prepare inputs for predictor (z only, no time injection)
        # We strict to the 'orange' run logic which was stable. 
        # Time conditioning polluted the representation learning when lambda was low.
        
        # b) Predictive loss removed as per user request.
        # We rely on Invariance (t0 augmentations) and SIGReg to shape the space.
        loss_pred = torch.tensor(0.0, device=self.device)

        # 3) Regularization: SIGReg or VICReg
        if self.reg_type == "vicreg":
            reg_result = self.regularizer(z_all, global_step=self.global_step, return_components=True)
            loss_sigreg = reg_result["total"]
            # Log VICReg components for diagnostics
            self.log(f"{mode}/vicreg_var_loss", reg_result["var_loss"])
            self.log(f"{mode}/vicreg_cov_loss", reg_result["cov_loss"])
            self.log(f"{mode}/vicreg_std_mean", reg_result["std_mean"])
            self.log(f"{mode}/vicreg_std_min", reg_result["std_min"])
        else:
            loss_sigreg = self.regularizer(z_all, global_step=self.global_step)

        # 4) Supervised forecast: decode from t0 backbone
        # Get t0 backbone embeddings [B, D] or [B, T, D]
        if emb_all.dim() == 3:
            emb_views = emb_all.view(B, V, -1, self.backbone_out)
            # Average over repeated t0 views, then pool over time.
            emb_t0 = emb_views[:, 1:-1, :, :].mean(dim=1)  # [B, T, D]
            pooled_t0 = emb_t0.mean(dim=1)                 # [B, D]
        else:
            emb_views = emb_all.view(B, V, -1)
            pooled_t0 = emb_views[:, 1:-1, :].mean(dim=1)  # [B, D]

        # Forecast with optional future covariate conditioning
        if self.use_covariates and future_times is not None:
            future_cov_emb = self.future_cov_encoder(future_times)  # [B, 64]
            forecast_input = torch.cat([pooled_t0.detach(), future_cov_emb], dim=-1)
        else:
            forecast_input = pooled_t0.detach()

        # NOTE: `detach()` makes the forecasting head a pure probe: forecast loss does not update
        # the encoder. This is intentional as we want to evaluate the representation quality.

        # NOTE: `-1` should equal B here. Using `view(B, ...)` is safer if shapes ever change.
        y_pred = self.forecast_head(forecast_input).view(B,self.input_dim, self.horizon)
        
        # Ensure y_true is [B, C, L] for loss calculation
        if y_true.shape[1] == self.horizon and y_true.shape[2] == self.input_dim:
            y_true = y_true.transpose(1, 2)
            
        loss_forecast = F.mse_loss(y_pred, y_true)
        
        loss_ssl = (loss_pred + temp_inv_loss) * (1 - self.lamb) + self.lamb * loss_sigreg
        total_loss = loss_forecast*0.1 + loss_ssl

        # logs
        self.log(f"{mode}/total_loss", total_loss, prog_bar=True)
        self.log(f"{mode}/mse_forecast", loss_forecast)
        self.log(f"{mode}/loss_ssl", loss_ssl)
        self.log(f"{mode}/pred_loss", loss_pred)
        self.log(f"{mode}/temp_inv_loss", temp_inv_loss)
        self.log(f"{mode}/sigreg", loss_sigreg)

        # Unscaled MAE for validation/test
        if mode in ["val", "test"] and self.scaler_mean is not None and self.scaler_std is not None:
            # NOTE: y_pred/y_true are [B, C, H] at this point.
            # TorchStandardScaler stores mean/std as [1, C, 1]; broadcasting relies on that.
            y_pred_unscaled = y_pred * self.scaler_std + self.scaler_mean
            y_true_unscaled = y_true * self.scaler_std + self.scaler_mean
            mae_unscaled = F.l1_loss(y_pred_unscaled, y_true_unscaled)
            self.log(f"{mode}/mae_unscaled", mae_unscaled, prog_bar=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode="val")

    def configure_optimizers(self):
        # Group 1: Self-Supervised components (Backbone, Projector, Predictor)
        ssl_params = (
            list(self.backbone.parameters()) + 
            list(self.proj.parameters()) + 
            list(self.predictor.parameters())
        )
        
        # Group 2: Forecasting Head (The "Probe")
        forecast_params = list(self.forecast_head.parameters())

        opt = torch.optim.AdamW([
            {"params": ssl_params, "lr": self.lr, "weight_decay": 5e-2},
            {"params": forecast_params, "lr": 1e-3, "weight_decay": 1e-7}
        ])
        

        # OneCycle usage: ensure trainer.estimated_stepping_batches is available
        # if hasattr(self.trainer, "estimated_stepping_batches") and self.trainer.estimated_stepping_batches is not None:
        #     sched = torch.optim.lr_scheduler.OneCycleLR(...)
        
        # We restore CosineAnnealingLR which was present in 'Blue' run equivalents (implied) or beneficial for convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, 
            T_max=self.trainer.max_epochs, 
            eta_min=1e-5
        )
        return {"optimizer": opt, "lr_scheduler": scheduler}

    @property
    def scaler(self):
        """Returns an object with inverse_transform method using stored mean/std."""
        if not hasattr(self, "scaler_mean") or self.scaler_mean is None:
            return None
        
        class SimpleScaler:
            def __init__(self, mean, std):
                self.mean = mean
                self.std = std
            def inverse_transform(self, x):
                # x shape: [B, C, L] or [B, L, C]
                # self.mean shape: [1, C, 1]
                if x.shape[1] == self.mean.shape[1]: # [B, C, L]
                    return x * self.std + self.mean
                # Fallback for [B, L, C] if needed
                return x * self.std.transpose(1, 2) + self.mean.transpose(1, 2)

        return SimpleScaler(self.scaler_mean, self.scaler_std)

    def probe(self, emb, future_times=None):
        """Standard interface for forecasting probe used by callbacks."""
        B = emb.shape[0]
        
        if self.use_covariates and future_times is not None:
            future_cov_emb = self.future_cov_encoder(future_times)
            forecast_input = torch.cat([emb, future_cov_emb], dim=-1)
        else:
            forecast_input = emb
            
        yhat = self.forecast_head(forecast_input)
        return yhat.view(B, self.input_dim, self.horizon)
