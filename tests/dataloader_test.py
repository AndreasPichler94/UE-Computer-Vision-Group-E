import unittest

from utils.aos_loader import _get_dataloader


class TestDataLoader(unittest.TestCase):
    def test_train_loader(self):
        data_loader = _get_dataloader("../data/train", focal_height=1.0, image_resolution=(128, 128))


