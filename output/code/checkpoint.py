import torch
import pickle
import os
import sys
from datetime import datetime


def get_checkpoint(model, optimizer, cuda_available, checkpoint_dir="./weights"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pth')]

    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        if cuda_available:
            checkpoint = torch.load(latest_checkpoint)
        else:
            checkpoint = torch.load(latest_checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_index = checkpoint['current_index']
        current_batch = checkpoint['current_batch']
        print(
            f"Checkpoint loaded from {latest_checkpoint}. Current index: {current_index}, Current epoch: {current_batch}")
        return current_index, current_batch
    else:
        print("No checkpoints found. Training from scratch.")
        return 0, 0


def save_checkpoint(model, optimizer, current_index, current_batch, checkpoint_dir='./weights'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{timestamp}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'current_index': current_index,
        'current_batch': current_batch,
    }, checkpoint_path)

    data = {"current_index": current_index, "current_batch": current_batch, "timestamp": timestamp}
    with open(os.path.join(checkpoint_dir, f'checkpoint_info_{timestamp}.pickle'), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Beispielaufruf:
# current_index, current_batch = get_checkpoint(model, optimizer)
# save_checkpoint(model, optimizer, current_index, current_batch)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # # check if checkpoint is compatible with the model
    # print("Checkpoint Architecture Shapes:")
    # for key, value in checkpoint['model_state_dict'].items():
    #     print(f"Layer: {key}, Shape: {value.shape}")
    #
    # # Print model architecture shapes
    # print("\nModel Architecture Shapes:")
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name}, Shape: {param.shape}")
    #
    # # Ensure the model architecture matches
    # assert checkpoint['model_architecture'] == model.architecture, "Model architecture mismatch"

    # Optional: Check other model properties

    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        # Ensure the optimizer is compatible
        assert checkpoint['optimizer_type'] == type(optimizer).__name__, "Optimizer type mismatch"
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer