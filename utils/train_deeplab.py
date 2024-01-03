import sys
import torch

from aos_loader import _get_dataloader
from evaluate import evaluate_model
sys.path.append("./models")
from unet_2.unet_model import UNet, UNetSmall


def check_gpu_availability():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return True
    else:
        return False


def train_deeplab(model, num_epochs=10):
    device = torch.device("cuda" if check_gpu_availability() else "cpu")

    train_loader = _get_dataloader(
        "./data/train/",
        focal_heights=(
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
        ),
        image_resolution=(512, 512),
        batch_size=10,
    )
    test_loader = _get_dataloader(
        "./data/test/",
        focal_heights=(
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
        ),
        image_resolution=(512, 512),
        batch_size=10,
    )

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"Number of samples {len(train_loader)}")
        print(f"Training epoch {epoch}")
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            print("Calculating zero_grad")
            model.optimizer.zero_grad()

            print("Calculating outputs")
            outputs = model(inputs)

            print("Calculating loss")

            if model.model_name == "UNet":
                rounded = torch.round(labels).squeeze(1).long()
                loss = model.criterion(outputs, rounded)
            else:
                loss = model.criterion(outputs["out"], labels.squeeze(1).long())

            print("Backprop")
            loss.backward()
            model.optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {running_loss/len(train_loader)}")

        with torch.no_grad():
            valid_loss = 0.0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                if model.model_name == "UNet" or model.model_name == "Deeplab":
                    rounded = torch.round(labels).squeeze(1).long()
                    loss = model.criterion(outputs, rounded)
                else:
                    loss = model.criterion(outputs["out"], labels.squeeze(1).long())

                valid_loss += loss.item()
            print(f"Validation Loss: {valid_loss/len(test_loader)}")

        # torch.save(model.state_dict(), "aosdeeplab_model.pth")

    # evaluate_model(model, train_loader, torch.nn.CrossEntropyLoss())


if __name__ == "__main__":
    import sys

    sys.path.append("./models")
    sys.path.append("./utils")

    import torch

    from aos_deeplab import AosDeepLab

    model = AosDeepLab(10, 2)
    # model = UNetSmall(10, 2)
    print(f"GPU available: {check_gpu_availability()}")

    trained_model = train_deeplab(model, num_epochs=10)

    # torch.save(trained_model.state_dict(), "aosdeeplab_model.pth")
