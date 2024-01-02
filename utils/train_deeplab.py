import sys
import torch

from aos_loader import _get_dataloader
from evaluate import evaluate_model


def check_gpu_availability():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return True
    else:
        return False


def train_deeplab(model, num_epochs=10):

    train_loader = _get_dataloader(
        "./data/train/", focal_height=0, image_resolution=(512, 512), batch_size=1
    )
    test_loader = _get_dataloader(
        "./data/test/", focal_height=0, image_resolution=(512, 512), batch_size=1
    )

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            model.optimizer.zero_grad()

            outputs = model(inputs)
            loss = model.criterion(outputs, labels)

            loss.backward()
            model.optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {running_loss/len(train_loader)}")
    
        with torch.no_grad():
            valid_loss = 0.0
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = model.criterion(outputs, labels)
                valid_loss += loss.item()
            print(f"Validation Loss: {valid_loss/len(test_loader)}")
        
        # torch.save(model.state_dict(), "aosdeeplab_model.pth")
        
    evaluate_model(model, train_loader, torch.nn.CrossEntropyLoss())
