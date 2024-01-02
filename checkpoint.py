import torch
import pickle
import os
from datetime import datetime


def get_checkpoint(model, optimizer, checkpoint_dir="./checkpoints"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pth')]

    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_index = checkpoint['current_index']
        current_batch = checkpoint['current_batch']
        print(
            f"Checkpoint loaded from {latest_checkpoint}. Current index: {current_index}, Current batch: {current_batch}")
        return current_index, current_batch
    else:
        print("No checkpoints found. Training from scratch.")
        return 0, 0


def save_checkpoint(model, optimizer, current_index, current_batch, checkpoint_dir='./checkpoints'):
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
