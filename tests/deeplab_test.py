import unittest
import torch
import sys

sys.path.append("../models")
from models.aos_deeplab import AosDeepLab


class TestDeeplab(unittest.TestCase):
    def test_deeplab_output(self):
        model = AosDeepLab()
        dummy_input = torch.rand(10, 10, 512, 512)
        output = model(dummy_input)

        self.assertIn("out", output)
        self.assertTrue(isinstance(output["out"], torch.Tensor))
        self.assertEqual(output["out"].shape, (10, 2, 512, 512))
