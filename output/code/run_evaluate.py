import torch

from models.aos_deeplab import AosDeepLab
from models.unet_2.unet_model import UNetSmall
from utils.aos_loader import _get_dataloader
from utils.evaluate import evaluation

focal_heights = (
    "0",
    "-0.4",
    "-0.8"
)
test_loader ,  _ = _get_dataloader("./data/test", focal_heights=focal_heights, image_resolution=(512, 512))
#model = UNet(10, 2)
model = AosDeepLab(len(focal_heights), 1, pixel_out=True)
evaluation("./checkpoints", model, test_loader, torch.nn.MSELoss())