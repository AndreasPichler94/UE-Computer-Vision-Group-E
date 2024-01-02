import torch
import torch.nn as nn
import torch.nn.functional as F
#import sys
#sys.path.append("..")

from models.unet import UNet

 # Function to test the model
def test_unet():
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Model parameters
    n_channels = 12  # Number of input channels
    n_classes = 1   # Number of output channels
    input_size = (128, 128)  # Input image size

    # Create the model
    model = UNet(n_channels, n_classes)

    # Generate random input data
    batch_size = 1
    random_input = torch.randn((batch_size, n_channels, input_size[0], input_size[1]))

    # Forward pass
    output = model(random_input)

    # Print output shape
    print("Input shape:", random_input.shape)
    print("Output shape:", output.shape)

# Test the U-Net model
test_unet()  

   


#Debuging
"""
from utils.aos_loader import _get_dataloader
from utils.evaluate import evaluate_model

if __name__ == "__main__":
    data_loader = _get_dataloader("./data/train/", focal_height=0, image_resolution=(128, 128), batch_size=1)
    evaluate_model(model, data_loader, nn.CrossEntropyLoss())

"""
