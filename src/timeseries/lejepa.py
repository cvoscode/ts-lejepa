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
                 proj_dim=128, num_slices=1024, lamb=0.5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder_backbone'])
        self.backbone = encoder_backbone                 # encoder should accept windows and return [B, T, D] or [B, D]
        self.backbone_out = getattr(encoder_backbone, "output_dim", None)
        if self.backbone_out is None:
            raise ValueError("encoder_backbone must expose .output_dim (dim of embedding per token / pooled).")

        # projector: map backbone outputs to proj_dim (use LayerNorm-friendly MLP)
        # design: if backbone returns a sequence [B, T, D] we will project per time-step
        self.proj = nn.Sequential(
            nn.LayerNorm(self.backbone_out),
            nn.Linear(self.backbone_out, 512),
            nn.GELU(),
            nn.Linear(512, proj_dim)
        )

        # predictor: map context projected embeddings -> target projected embeddings
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, 512),
            nn.GELU(),
            nn.Linear(512, proj_dim)
        )

        # SIGReg: prefer using lejepa implementation if available
        if have_lejepa:
            # example usage: SlicingUnivariateTest(univariate_test=EppsPulley(...), num_slices=num_slices)
            from lejepa.univariate import EppsPulley
            self.sigreg = SlicingUnivariateTest(univariate_test=EppsPulley(num_points=17), num_slices=num_slices)
        else:
            self.sigreg = TemporalSIGReg(feature_dim=proj_dim, num_slices=num_slices, A_dim=256)

        # forecasting head: map pooled context embedding to horizon values (you may want to use autoregressive)
        self.forecast_head = nn.Sequential(
            nn.Linear(self.backbone_out, 256),
            nn.GELU(),
            nn.Linear(256, horizon * input_dim)
        )
        self.horizon = horizon
        self.input_dim = input_dim
        self.lr = lr
        self.lamb = lamb

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
          context_window: [B, ...] (past)
          target_window:  [B, ...] (future window we want to predict)
          y_true: actual future numeric targets used for supervised MSE (optional)
        """
        context_windows, target_windows, y_true = batch

        # 1) encode context and target with same encoder (shared weights)
        emb_context = self.encode_window(context_windows)  # [B, T_c, D] or [B, D]
        emb_target  = self.encode_window(target_windows)   # [B, T_t, D] or [B, D]

        # project to JEPA latent per time-step (if seq) or single pooled vector
        # if emb_* is [B, T, D] then project per time-step -> [B, T, P]
        def project(emb):
            if emb.dim() == 3:
                B, T, D = emb.shape
                emb_flat = emb.reshape(-1, D)
                z_flat = self.proj(emb_flat)            # [B*T, P]
                return z_flat.view(B, T, -1)           # [B, T, P]
            else:
                return self.proj(emb)                  # [B, P]

        z_context = project(emb_context)
        z_target  = project(emb_target)

        # 2) JEPA predictive loss: predict target proj from context proj.
        # Strategy: pool context (mean/attn or last token) and predict pooled target, or predict sequence -> I show pooled strategy for clarity.
        if z_context.dim() == 3:
            pooled_ctx = z_context.mean(dim=1)   # [B, P]  (alternative: attentive pooling)
        else:
            pooled_ctx = z_context               # [B, P]

        if z_target.dim() == 3:
            pooled_tgt = z_target.mean(dim=1)    # [B, P]
        else:
            pooled_tgt = z_target                # [B, P]

        pred_tgt = self.predictor(pooled_ctx)    # [B, P]
        loss_pred = F.mse_loss(pred_tgt, pooled_tgt)

        # 3) SIGReg regularization: apply to the *set* of embeddings we want to regularize.
        # Combine many embeddings: stack pooled targets (or all projected vectors)
        if z_target.dim() == 3:
            sig_input = z_target.reshape(-1, z_target.size(-1))  # [B*T, P]
        else:
            sig_input = pooled_tgt                               # [B, P]

        # if using lejepa package, the API accepts embeddings [N, D]; else our TemporalSIGReg also accepts [N, D]
        loss_sigreg = self.sigreg(sig_input)

        # 4) supervised forecast (if y_true provided). decode from pooled context or use sequence decoder.
        # Here we decode from pooled backbone (emb_context could be pooled from encoder output)
        if emb_context.dim() == 3:
            pooled_backbone = emb_context.mean(dim=1)
        else:
            pooled_backbone = emb_context

        y_pred = self.forecast_head(pooled_backbone).view(-1, self.horizon, self.input_dim)
        loss_forecast = F.mse_loss(y_pred, y_true)
        loss_ssl = loss_pred*(1 - self.lamb) + self.lamb * loss_sigreg
        total_loss = loss_forecast + 0.1 * loss_ssl

        # logs
        self.log(f"{mode}/total_loss", total_loss, prog_bar=True)
        self.log(f"{mode}/mse_forecast", loss_forecast)
        self.log(f"{mode}/jepar_pred", loss_pred)
        self.log(f"{mode}/sigreg", loss_sigreg)

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
