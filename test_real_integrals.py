import sys

import numpy as np
import torch
from PIL import Image
import os
from torchvision import transforms
sys.path.append("./utils")
from utils import train_deeplab


def test_with_real_integrals(model, folder):
    device = torch.device("cuda" if train_deeplab.check_gpu_availability() else "cpu")
    res = (512, 512)
    focal_points = [
        "0.00",
        "0.20",
        "0.40",
        "0.60",
        "0.80",
        "1.00",
        "1.20",
        "1.60",
        "2.00",
        "2.40",
    ]
    model.to(device)

    image_names = [os.path.join(folder, f"FP_{fp}.png") for fp in focal_points]

    to_tensor = transforms.ToTensor()

    images = [to_tensor(Image.open(img_name).convert("L").resize(res)) for img_name in image_names]
    

    input_tensor = torch.cat(images, dim=0).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)

    train_deeplab.visualize_tensors(
        "visualization_real_integral",
        0,
        0,
        input_tensor=input_tensor[0][0],
        prediction_tensor=outputs["out"][0],
        target_tensor=input_tensor[0][0],
        ground_truth=input_tensor[0][0],
    )


if __name__ == "__main__":
    import sys

    sys.path.append("./models")
    sys.path.append("./utils")

    from models.aos_deeplab import AosDeepLab
    from checkpoint import get_checkpoint

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

    model = AosDeepLab(len(focal_heights), 1, pixel_out=True)    

    iteration, epoch = get_checkpoint(model, model.optimizer)
    test_with_real_integrals(model, "./focal_stack/")