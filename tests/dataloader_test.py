import unittest

import torch

from utils.aos_loader import _get_dataloader


class TestDataLoader(unittest.TestCase):
    def test_train_loader(self):
        data_loader = _get_dataloader("../data/train", focal_heights=("0", "0"), image_resolution=(128, 128), batch_size=1)
        for (x, y) in data_loader:
            self.assertEqual(type(x), torch.Tensor)
            self.assertEqual(type(y), torch.Tensor)



