# changeset: improved TemporalSIGReg + JEPA training loop
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchvision.ops import MLP

# optional: prefer using official lejepa if installed (recommended)
try:
    from lejepa.multivariate import SlicingUnivariateTest
    have_lejepa = True
except Exception:
    have_lejepa = False
print(f"LeJEPA available: {have_lejepa}")
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

        # create the univariate test weights (same as LeJEPA: t grid, window, weights)
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)                 # [K]
        self.register_buffer("phi", window)          # [K]
        self.register_buffer("weights", weights * window)  # [K]

        # pre-draw random projection matrix of shape [D, num_slices]
        # we store them as a buffer so computations are deterministic (but cheap to replace)
        rng = torch.Generator()
        rng.manual_seed(seed)
        A = torch.randn(feature_dim, num_slices, generator=rng)
        # normalize columns (each slice is a unit vector)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-12)
        self.register_buffer("A", A)   # [D, S]

    def forward(self, z):
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
        proj = z_flat @ self.A        # [N, S]

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
                 scaler_mean=None, scaler_std=None):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder_backbone'])
        self.backbone = encoder_backbone                 # encoder should accept windows and return [B, T, D] or [B, D]
        self.backbone_out = getattr(encoder_backbone, "output_dim", None)
        if self.backbone_out is None:
            raise ValueError("encoder_backbone must expose .output_dim")

        # Projector: map backbone outputs to proj_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(self.backbone_out),
            nn.Linear(self.backbone_out, 512),
            nn.GELU(),
            nn.Linear(512, proj_dim)
        )

        # Predictor: map context projected embeddings -> target projected embeddings
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, 512),
            nn.GELU(),
            nn.Linear(512, proj_dim)
        )

        # SIGReg Setup
        if have_lejepa:
            from lejepa.univariate import EppsPulley
            self.sigreg = SlicingUnivariateTest(univariate_test=EppsPulley(num_points=17), num_slices=num_slices)
        else:
            self.sigreg = TemporalSIGReg(feature_dim=proj_dim, num_slices=num_slices, A_dim=256)

        # Forecasting head
        self.forecast_head = nn.Sequential(
            nn.Linear(self.backbone_out, 256),
            nn.GELU(),
            nn.Linear(256, input_dim*horizon)
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

    def forward(self, x):
        """Standard forward pass: encode a single window or multiple views."""
        # If x is [B, C, L], just encode it
        # If x is [B, V, C, L], we typically want to encode the current view (V=1) or mean of views
        if x.ndim == 4:
            B, V, C, L = x.shape
            x = x.view(B * V, C, L)
            emb = self.backbone(x)
            return emb.view(B, V, -1)
        return self.backbone(x)

    def encode_window(self, window):
        """
        window: [B, C, T] or [B, T, C] depending on your backbone
        returns: embeddings per token/time-step [B, T, D] OR pooled [B, D]
        (maintain consistent shape for your backbone)
        """
        emb = self.backbone(window)  # assume [B, T, D] or [B, D]
        return emb

    def shared_step(self, batch, batch_idx, mode="train"):
        """
        Expect dataset to return:
          views: [B, V, C, L] (in our case V=3: t-1, t0, t+1)
          y_true: [B, C, L_target] (actual future numeric targets)
        """
        views, y_true = batch
        B, V, C, L = views.shape

        # 1) Encode all views
        # views: [B, V, C, L] -> flatten batch and view dimensions for encoder
        views_flat = views.view(B * V, C, L)
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
        z_prev = pooled_views[:, 0, :]
        z_curr = pooled_views[:, 1, :]
        z_next = pooled_views[:, 2, :]

        # 2) JEPA Loss: 
        # a) Consistency loss: t-1, t0, t+1 should be similar in embedding space
        # We can use MSE or SIGReg on these.
        temp_inv_loss = (F.mse_loss(z_prev, z_curr) + F.mse_loss(z_next, z_curr)) * 0.5
        
        # b) Predictive loss: predict "something" from t0. 
        # In JEPA, we usually predict a target representation. 
        # Since we don't have a "target" window in the input views (except y_true which is raw),
        # we can either:
        # 1. Predict z_curr from z_prev? 
        # 2. Predict a future embedding if we had one.
        # User goal: "get the model to embedd t-1,t0,t+1 at a similar position"
        # So consistency loss is key.
        
        pred_curr = self.predictor(z_prev)
        pred_next = self.predictor(z_curr)
        loss_pred = F.mse_loss(pred_curr, z_curr)
        loss_pred += F.mse_loss(pred_next, z_next)

        # 3) SIGReg regularization: apply to the *set* of embeddings
        loss_sigreg = self.sigreg(z_all)

        # 4) Supervised forecast: decode from t0 backbone
        # Get t0 backbone embeddings [B, D] or [B, T, D]
        if emb_all.dim() == 3:
            emb_t0 = emb_all.view(B, V, -1, self.backbone_out)[:, 1, :, :]
            pooled_t0 = emb_t0.mean(dim=1)
        else:
            emb_t0 = emb_all.view(B, V, -1)[:, 1, :]
            pooled_t0 = emb_t0

        y_pred = self.forecast_head(pooled_t0).view(-1, self.input_dim, self.horizon)
        
        # Ensure y_true is [B, C, L] for loss calculation
        if y_true.shape[1] == self.horizon and y_true.shape[2] == self.input_dim:
            y_true = y_true.transpose(1, 2)
            
        loss_forecast = F.mse_loss(y_pred, y_true)
        
        loss_ssl = (loss_pred + temp_inv_loss) * (1 - self.lamb) + self.lamb * loss_sigreg
        total_loss = loss_forecast + 0.1 * loss_ssl

        # logs
        self.log(f"{mode}/total_loss", total_loss, prog_bar=True)
        self.log(f"{mode}/mse_forecast", loss_forecast)
        self.log(f"{mode}/pred_loss", loss_pred)
        self.log(f"{mode}/temp_inv_loss", temp_inv_loss)
        self.log(f"{mode}/sigreg", loss_sigreg)

        # Unscaled MAE for validation/test
        if mode in ["val", "test"] and self.scaler_mean is not None and self.scaler_std is not None:
            # y_pred, y_true are [B, H, C]
            # self.scaler_mean/std are [1, 1, C]
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
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        # OneCycle usage: ensure trainer.estimated_stepping_batches is available at configure time
        if hasattr(self.trainer, "estimated_stepping_batches") and self.trainer.estimated_stepping_batches is not None:
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr,
                                                       total_steps=self.trainer.estimated_stepping_batches)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        return opt

    @property
    def scaler(self):
        """Returns an object with inverse_transform method using stored mean/std."""
        if not hasattr(self, "scaler_mean"):
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

    def probe(self, emb):
        """Standard interface for forecasting probe used by callbacks."""
        B = emb.shape[0]
        yhat = self.forecast_head(emb)
        return yhat.view(B, self.input_dim, self.horizon)
