import torch
import torchvision
import os
import onnx
import sys 
sys.path.append("./models")
from aos_deeplab import AosDeepLab
sys.path.append("./utils")
from checkpoint import get_checkpoint

def check_gpu_availability():
    return torch.cuda.is_available()

n_channels = 10 
n_classes = 2 

model = AosDeepLab(n_channels, n_classes)
optimizer = model.optimizer  # Using the optimizer defined within the model class

# Load the checkpoint
current_index, current_batch = get_checkpoint(model, optimizer)

# Set the model to evaluation mode before exporting
model.eval()

# Create a dummy variable with the correct shape
# The shape depends on the input your model expects.
# For example, if you're working with RGB images of 224x224, the dummy input would be:
dummy_input = torch.randn(1, n_channels, 512, 512)  # Update shape if necessary

# Export the model to ONNX format
torch.onnx.export(model,               # model being run
                  dummy_input,         # model input (or a tuple for multiple inputs)
                  "aosdeeplab_model.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,    # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

print("AosDeepLab model has been converted to ONNX format and saved as aosdeeplab_model.onnx")