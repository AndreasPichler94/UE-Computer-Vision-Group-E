import torch
import pickle
import os


def get_checkpoint(model, optimizer, checkpoint_dir="./checkpoints"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_index = checkpoint['current_index']
        current_batch = checkpoint['current_batch']
        print(f"Checkpoint loaded. Current index: {current_index}, Current batch: {current_batch}")
        return current_index, current_batch
    else:
        print("No checkpoint found. Training from scratch.")
        return 0, 0


def save_checkpoint(model, optimizer, current_index, current_batch, checkpoint_dir='./checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'current_index': current_index,
        'current_batch': current_batch,
    }, checkpoint_path)

    data = {"current_index": current_index, "current_batch": current_batch}
    with open(os.path.join(checkpoint_dir, 'checkpoint_info.pickle'), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)