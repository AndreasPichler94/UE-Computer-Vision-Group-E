import sys

import numpy as np
import torch
from PIL import Image
import os
import re
from torchvision import transforms
sys.path.append("./utils")
from train import check_gpu_availability, visualize_tensors


def get_batch_prefixes(folder):
    file_names = os.listdir(folder)
    prefixes = set(re.match(r'(\d+_\d+)_', name).group(1) for name in file_names if re.match(r'\d+_\d+_', name))
    return prefixes


def test(model, folder):
    device = torch.device("cuda" if check_gpu_availability() else "cpu")
    res = (512, 512)
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

    model.to(device)

    to_tensor = transforms.ToTensor()

    for batch_prefix in get_batch_prefixes(folder):
        # Identify ground truth and focal stack images
        ground_truth_name = os.path.join(folder, f"{batch_prefix}_GT_pose_0_thermal.png")
        focal_stack_names = [os.path.join(folder, f"{batch_prefix}-aos_thermal-{fh}.png") for fh in focal_heights]

        images = [to_tensor(Image.open(img_name).convert("L").resize(res)) for img_name in focal_stack_names]
        ground_truth_image = to_tensor(Image.open(ground_truth_name).convert("L").resize(res))

        input_tensor = torch.cat(images, dim=0).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)

        visualize_tensors(
            f"Result folder/{focal_stack_names[0].split('/')[1].split('--')[0]}",
            0,
            0,  
            input_tensor=input_tensor[0][0],
            prediction_tensor=outputs["out"][0],
            target_tensor=ground_truth_image, 
            ground_truth=ground_truth_image,
        )
        print(f"Generated result image in ./Result folder/{focal_stack_names[0].split('/')[1].split('--')[0]}")


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

    iteration, epoch = get_checkpoint(model, model.optimizer, check_gpu_availability())
    test(model, "test data folder/")