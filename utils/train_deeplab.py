import sys
import torch

from aos_loader import _get_dataloader
from evaluate import evaluate_model
sys.path.append("./models")
from unet_2.unet_model import UNet, UNetSmall
sys.path.append(".")
from checkpoint import save_checkpoint, get_checkpoint
import matplotlib.pyplot as plt


def check_gpu_availability():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return True
    else:
        return False


def train_deeplab(model, num_epochs=10, current_index=0, current_epoch=0):
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
        batch_size=15,
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
        batch_size=15,
    )

    model.to(device)
    for state in model.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)


    for epoch in range(current_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        print(f"Number of samples {len(train_loader)}")
        print(f"Training epoch {epoch}")

        for ind, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            if model.model_name == "Deeplab":
                threshold = 0.7843
                labels = torch.where(labels > threshold, 
                     torch.tensor(1.0, device=labels.device, dtype=torch.long), 
                     torch.tensor(0.0, device=labels.device, dtype=torch.long))
                
            print(f"Iteration {ind}")
            model.optimizer.zero_grad()
            outputs = model(inputs)

            if model.model_name == "UNet":
                rounding_threshold = 0.8 # round to 0 below this value
                rounded = (labels > rounding_threshold).squeeze(1).long()

                if ind % 100 == 0:
                    print("Showing network outputs...")
                    visualize_tensors(ind, input_tensor=inputs[0][0], prediction_tensor=torch.argmax(outputs[0], dim=0, keepdim=True), target_tensor=rounded[0], ground_truth=labels[0])

                loss = model.criterion(outputs, rounded)
            elif model.model_name == "Deeplab":
                if ind % 100 == 0:
                    # rounded = torch.where(labels >= 200, torch.tensor(1, device=labels.device), torch.tensor(0, device=labels.device))
                    print("Showing network outputs...")
                    visualize_tensors(ind, input_tensor=inputs[0][0], prediction_tensor=torch.argmax(outputs["out"][0], dim=0, keepdim=True), target_tensor=labels[0], ground_truth=labels[0])

                loss = model.criterion(outputs["out"], labels.squeeze(1).long())
            else:
                loss = model.criterion(outputs["out"], rounded)

            loss.backward()
            model.optimizer.step()

            running_loss += loss.item()

            if ind % 1000 == 0:
                print(f"Iteration {ind}, Loss: {running_loss/len(train_loader)}")
                save_checkpoint(model, model.optimizer, ind, epoch, checkpoint_dir='./checkpoints')

        print(f"Epoch {epoch}, Loss: {running_loss/len(train_loader)}")

        with torch.no_grad():
            valid_loss = 0.0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                if model.model_name == "UNet":
                    rounded = torch.round(labels).squeeze(1).long()
                    loss = model.criterion(outputs, rounded)
                else:
                    loss = model.criterion(outputs["out"], labels.squeeze(1).long())

                valid_loss += loss.item()
            print(f"Validation Loss: {valid_loss/len(test_loader)}")

    # evaluate_model(model, train_loader, torch.nn.CrossEntropyLoss())


def visualize_tensors(ind, input_tensor, prediction_tensor, target_tensor, ground_truth, cmap='gray'):
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
    # plt.show()
    plt.savefig(f"visualization_frame{ind}.png")


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

    iteration, epoch = get_checkpoint(model, model.optimizer)

    trained_model = train_deeplab(model, num_epochs=10, current_epoch=epoch, current_index=iteration)

    # torch.save(trained_model.state_dict(), "aosdeeplab_model.pth")
