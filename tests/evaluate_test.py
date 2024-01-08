import unittest
import torch

from checkpoint import load_checkpoint
from models.Unet import UNetSmall, UNet
from models.aos_deeplab import AosDeepLab
from utils.aos_loader import _get_dataloader
from utils.evaluate import evaluate_model, evaluation

class TestEvaluateModel(unittest.TestCase):
    # How to use this test:
    # insert correct path to your checkpoint file
    # create equal model as in checkpoint file
    # use correct number of focal heights as in checkpoint file
    # check if you have correct path to your test data
    # run test
    def test_evaluate_model(self):
        checkpoint_path = "../checkpoints/checkpoint_20240105_185338.pth"

        #model = UNetSmall(10, 2)
        model = AosDeepLab(10, 2)
        model, optimiser = load_checkpoint(checkpoint_path, model, optimizer=None)

        criterion = torch.nn.CrossEntropyLoss()
        focal_heights = (
            "0",
            "-0.2",
            "-0.4",
            "-0.6",
            "-0.8",
            "-1.0",
            "-1.2",
            "-1.6",
            "-2.0",
            "-2.4",
        )

        test_loader = _get_dataloader("../data/test", focal_heights=focal_heights, image_resolution=(128, 128))

        evaluate_model(model, test_loader, criterion)

    # How to use this test:
    # insert correct path to your checkpoint files
    # create equal model as in checkpoint files
    # use correct number of focal heights as in checkpoint files
    # check if you have correct path to your test data
    # run test
    def test_evaluation(self):
        focal_heights = (
            "0",
            "-0.2",
            "-0.4",
            "-0.6",
            "-0.8",
            "-1.0",
            "-1.2",
            "-1.6",
            "-2.0",
            "-2.4",
        )
        test_loader = _get_dataloader("../data/test", focal_heights=focal_heights, image_resolution=(128, 128))
        #model = UNet(10, 2)
        model = AosDeepLab(10, 2)
        evaluation("../checkpoints", model, test_loader, torch.nn.CrossEntropyLoss())
