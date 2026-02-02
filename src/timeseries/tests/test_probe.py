import unittest

import torch

from timeseries.tasks.forecast_probe import ForecastProbe


class TestForecastProbe(unittest.TestCase):
    def test_probe_output_shape(self):
        probe = ForecastProbe(input_dim=8, horizon=6, output_channels=3)
        emb = torch.randn(4, 8)
        out = probe(emb)
        self.assertEqual(out.shape, (4, 3, 6))


if __name__ == "__main__":
    unittest.main()
