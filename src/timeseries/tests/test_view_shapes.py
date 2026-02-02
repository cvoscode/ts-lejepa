import unittest

import torch

from timeseries.core.types import Batch


class TestViewShapes(unittest.TestCase):
    def test_batch_validate_views(self):
        views = torch.zeros(2, 3, 4, 5)
        view_times = torch.zeros(2, 3, 5, 6)
        batch = Batch(views=views, view_times=view_times)
        batch.validate()

    def test_batch_validate_mismatch(self):
        views = torch.zeros(2, 3, 4, 5)
        view_times = torch.zeros(2, 2, 5, 6)
        batch = Batch(views=views, view_times=view_times)
        with self.assertRaises(ValueError):
            batch.validate()


if __name__ == "__main__":
    unittest.main()
