
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from timeseries.core.types import Batch, InvarianceMix
from timeseries.ssl.lejepa import LeJEPA_SSL
from timeseries.tasks.forecast_probe import ForecastProbe
from timeseries.train.loop import run_ssl_train_loop


class DummyEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.output_mode = "pooled"
        self.linear = nn.Linear(8, output_dim)

    def forward(self, x: torch.Tensor, time_features=None) -> torch.Tensor:
        # x: [B, C, T] -> [B, D]
        if x.dim() == 4: # when flattened: [B*V, C, T]
            x = x.mean(dim=2) # [B*V, C]
        elif x.dim() == 3:
             x = x.mean(dim=2)
        
        # Assume input C=8 for this test
        # If C is not 8, we project it.
        if x.shape[1] != 8:
             # simple mean pad
             pass
        return self.linear(x)


class DummyProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TestRunLoop(unittest.TestCase):
    def test_run_loop_updates_probe(self):
        # Setup dims
        B, V, C, T = 4, 3, 8, 10
        H = 5
        D = 16
        P = 16

        # Models
        encoder = DummyEncoder(output_dim=D)
        projector = DummyProjector(D, P)
        ssl_core = LeJEPA_SSL(
            encoder=encoder,
            projector=projector,
            proj_dim=P,
            lamb=0.5
        )
        
        probe = ForecastProbe(
            input_dim=D,
            horizon=H,
            output_channels=C,
            use_covariates=False
        )

        # Clone initial weights related to probe head
        initial_weight = probe.head[1].weight.clone()

        # Create dummy batch with targets
        views = torch.randn(B, V, C, T)
        targets = torch.randn(B, H, C) # [B, H, C]
        batch = Batch(views=views, targets=targets)
        
        loader = [batch] # iterate once

        cfg = {
            "epochs": 1,
            "lr": 1e-2,
            "probe_weight": 1.0, 
            "weight_decay": 0.0
        }

        # Run loop
        history = run_ssl_train_loop(
            cfg=cfg,
            ssl_core=ssl_core,
            train_loader=loader,
            probe=probe,
            device=torch.device("cpu")
        )

        # Check that probe weights changed
        updated_weight = probe.head[1].weight
        diff = (initial_weight - updated_weight).abs().sum().item()
        
        self.assertGreater(diff, 1e-5, "Probe weights did not update during training loop")
        self.assertIn("train_ssl", history)

if __name__ == "__main__":
    unittest.main()
