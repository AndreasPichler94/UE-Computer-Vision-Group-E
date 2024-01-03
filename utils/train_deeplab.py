import sys
import torch

from aos_loader import _get_dataloader
from evaluate import evaluate_model
sys.path.append("./models")
from unet_2.unet_model import UNet, UNetSmall
import matplotlib.pyplot as plt


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
        for ind, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            print("Calculating zero_grad")
            model.optimizer.zero_grad()

            print("Calculating outputs")
            outputs = model(inputs)

            print("Calculating loss")

            if model.model_name == "UNet":
                rounding_threshold = 0.8 # round to 0 below this value
                rounded = (labels > rounding_threshold).squeeze(1).long()

                if ind % 100 == 0:
                    print("Showing network outputs...")
                    visualize_tensors(input_tensor=inputs[0][0], prediction_tensor=torch.argmax(outputs[0], dim=0, keepdim=True), target_tensor=rounded[0], ground_truth=labels[0])

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


def visualize_tensors(input_tensor, prediction_tensor, target_tensor, ground_truth, cmap='hot'):
    # Ensure the tensors are detached and moved to cpu
    input_tensor = input_tensor.detach().cpu()
    prediction_tensor = prediction_tensor.detach().cpu()
    target_tensor = target_tensor.detach().cpu()
    ground_truth = ground_truth.detach().cpu()

    # Create a subplot with 3 columns for the 3 images
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

    # Plot input_tensor
    axes[0].imshow(input_tensor.squeeze(), cmap=cmap)
    axes[0].set_title("input")

    # Plot prediction_tensor
    axes[1].imshow(prediction_tensor.squeeze(), cmap=cmap)
    axes[1].set_title("prediction")

    # Plot target_tensor
    axes[2].imshow(target_tensor.squeeze(), cmap=cmap)
    axes[2].set_title("target")

    # Plot ground_truth
    axes[3].imshow(ground_truth.squeeze(), cmap=cmap)
    axes[3].set_title("ground truth")

    # Display the plot
    plt.tight_layout()
    plt.show()


# Create some random data for example purpose
input_tensor = torch.rand((1, 512, 512))
prediction_tensor = torch.rand((1, 512, 512))
target_tensor = torch.rand((1, 512, 512))

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
