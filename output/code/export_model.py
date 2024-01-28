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
optimizer = model.optimizer  

# Load the checkpoint
current_index, current_batch = get_checkpoint(model, optimizer, check_gpu_availability())

# Set the model to evaluation mode before exporting
model.eval()


dummy_input = torch.randn(1, n_channels, 512, 512)  

torch.onnx.export(model,              
                  dummy_input,        
                  "aosdeeplab_model.onnx",  
                  export_params=True,  
                  opset_version=10,    
                  do_constant_folding=True, 
                  input_names = ['input'],   
                  output_names = ['output'], 
                  dynamic_axes={'input' : {0 : 'batch_size'},    
                                'output' : {0 : 'batch_size'}})

print("AosDeepLab model has been converted to ONNX format and saved as aosdeeplab_model.onnx")