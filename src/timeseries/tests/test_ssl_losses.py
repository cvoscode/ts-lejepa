import unittest

import torch
import torch.nn as nn

from timeseries.ssl.lejepa import LeJEPA_SSL


class DummyEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x.mean(dim=-1)
        return x


class DummyProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TestSSLLosses(unittest.TestCase):
    def test_invariance_loss_identical_views(self):
        encoder = DummyEncoder(output_dim=4)
        projector = DummyProjector(4, 4)
        ssl = LeJEPA_SSL(encoder=encoder, projector=projector, proj_dim=4, lamb=0.5)

        views = torch.ones(2, 3, 4, 5)
        res = ssl(views)
        self.assertLess(res.inv_loss.item(), 1e-6)

    def test_sigreg_accepts_token(self):
        encoder = DummyEncoder(output_dim=4)
        projector = DummyProjector(4, 4)
        ssl = LeJEPA_SSL(encoder=encoder, projector=projector, proj_dim=4, lamb=0.5)

        views = torch.randn(2, 3, 4, 5)
        res = ssl(views)
        self.assertTrue(torch.isfinite(res.sigreg_loss))


if __name__ == "__main__":
    unittest.main()
