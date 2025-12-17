import torch
class TorchStandardScaler:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.mean = None   # stored as [1, C, 1]
        self.std = None    # stored as [1, C, 1]

    def _ensure_3d(self, x: torch.Tensor):
        """Convert [T, C] → [1, C, T] or keep [N, C, L] unchanged."""
        if x.ndim == 2:
            # [T, C] → [1, C, T]
            return x.permute(1, 0).unsqueeze(0)
        elif x.ndim == 3:
            return x
        else:
            raise ValueError(f"Input must be 2D or 3D, got shape {x.shape}")

    def fit(self, x: torch.Tensor):
        x = self._ensure_3d(x)  # always [N, C, L]
        # mean/std over N and L → keep C
        self.mean = x.mean(dim=(0, 2), keepdim=True)  # [1, C, 1]
        self.std = x.std(dim=(0, 2), keepdim=True)    # [1, C, 1]
        

    def transform(self, x: torch.Tensor):
        x3 = self._ensure_3d(x)
        out = (x3 - self.mean) / (self.std + self.eps)
        return out

    def inverse_transform(self, x: torch.Tensor):
        x3 = self._ensure_3d(x)
        out = x3 * (self.std + self.eps) + self.mean
        return out

    def fit_transform(self, x: torch.Tensor):
        self.fit(x)
        return self.transform(x)

    def to(self, device):
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        return self
