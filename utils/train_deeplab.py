import sys
import torch

from aos_loader import _get_dataloader
from evaluate import evaluate_model
sys.path.append("./models")
from unet_2.unet_model import UNet, UNetSmall
sys.path.append(".")
from checkpoint import save_checkpoint, get_checkpoint
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


def check_gpu_availability():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return True
    else:
        return False


def train_deeplab(model, focal_heights, num_epochs=10, current_index=0, current_epoch=0):
    writer = SummaryWriter(f'trainlogs/deeplab_training_{num_epochs}_epochs')
    device = torch.device("cuda" if check_gpu_availability() else "cpu")

    batch_size = 15
    train_loader = _get_dataloader(
        "./data/train/",
        focal_heights=focal_heights,
        image_resolution=(512, 512),
        batch_size=batch_size,
    )
    test_loader = _get_dataloader(
        "./data/test/",
        focal_heights=focal_heights,
        image_resolution=(512, 512),
        batch_size=batch_size,
    )

    model.to(device)
    for state in model.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    rounding_threshold = 0.8  # round to 0 below this value
    for epoch in range(current_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        print(f"Number of samples {len(train_loader)}")
        print(f"Training epoch {epoch}")
        for ind, (inputs, labels) in tqdm(enumerate(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)

            if model.model_name == "Deeplab":
                threshold = 0.7843
                labels = torch.where(labels > threshold,
                                     torch.tensor(1.0, device=labels.device, dtype=torch.long),
                                     torch.tensor(0.0, device=labels.device, dtype=torch.long))

            model.optimizer.zero_grad()

            outputs = model(inputs)

            if model.model_name == "UNet":

                if model.pixel_out:
                    loss = model.criterion(outputs, labels)
                    target_tensor = None
                else:
                    rounded = (labels > rounding_threshold).squeeze(1).long()
                    loss = model.criterion(outputs, rounded)
                    target_tensor = rounded[0]

                if (ind - 5) % 100 == 0:
                    print(f"Showing network outputs... Loss: {running_loss / (ind + 1)}")
                    visualize_tensors(ind, input_tensor=inputs[0][0],
                                      prediction_tensor=torch.softmax(outputs[0], 0)[1, :, :],
                                      target_tensor=target_tensor, ground_truth=labels[0])
            elif model.model_name == "Deeplab":
                if ind % 100 == 0:
                    # rounded = torch.where(labels >= 200, torch.tensor(1, device=labels.device), torch.tensor(0, device=labels.device))
                    print("Showing network outputs...")
                    visualize_tensors(ind, input_tensor=inputs[0][0],
                                      prediction_tensor=torch.argmax(outputs["out"][0], dim=0, keepdim=True),
                                      target_tensor=labels[0], ground_truth=labels[0])

                loss = model.criterion(outputs["out"], labels.squeeze(1).long())
            else:
                raise NotImplementedError("Only UNet and Deeplab implemented")

            loss.backward()
            model.optimizer.step()

            running_loss += loss.item()

            if ind % 1000 == 0:
                print(f"Iteration {ind}, Loss: {running_loss/len(train_loader)}")
                save_checkpoint(model, model.optimizer, ind, epoch, checkpoint_dir='./checkpoints')
                writer.add_scalar('Loss/train', running_loss/(ind + 1), epoch * len(train_loader) + ind)

        print(f"Epoch {epoch}, Loss: {running_loss/len(train_loader)}")

        model.eval()
        valid_loss = 0.0
        total_iou = 0.0
        total_batches = 0
        with torch.no_grad():   
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                if model.model_name == "UNet":
                    rounded = (labels > rounding_threshold).squeeze(1).long()
                    loss = model.criterion(outputs, rounded)
                else:
                    loss = model.criterion(outputs["out"], labels.squeeze(1).long())

                valid_loss += loss.item()

                preds = outputs["out"] if model.model_name != "UNet" else outputs
                mean_iou = calculate_mean_iou(preds, labels.squeeze(1))
                total_iou += mean_iou
                total_batches += 1
            
            avg_valid_loss = valid_loss / total_batches
            avg_valid_iou = total_iou / total_batches
            print(f"Validation Loss: {avg_valid_loss}")
            print(f"Validation Mean IoU: {avg_valid_iou}")
            writer.add_scalar('Loss/validation', avg_valid_loss, epoch)
            writer.add_scalar('Mean IoU/validation', avg_valid_iou, epoch)

    # evaluate_model(model, train_loader, torch.nn.CrossEntropyLoss())
    writer.close()

def visualize_tensors(ind, input_tensor, prediction_tensor, target_tensor, ground_truth, cmap='gray'):
    # Ensure the tensors are detached and moved to cpu
    input_tensor = input_tensor.detach().cpu()
    prediction_tensor = prediction_tensor.detach().cpu()
    ground_truth = ground_truth.detach().cpu()

    # Create a subplot with 3 columns for the 3 images
    fig, axes = plt.subplots(nrows=1, ncols=(4 if target_tensor is not None else 3), figsize=(15 + (5 if target_tensor is not None else 0), 5))

    # Plot input_tensor
    axes[0].imshow(input_tensor.squeeze(), cmap=cmap)
    axes[0].set_title("input")

    # Plot prediction_tensor
    axes[1].imshow(prediction_tensor.squeeze(), cmap=cmap)
    axes[1].set_title("prediction")

    # Plot ground_truth
    axes[2].imshow(ground_truth.squeeze(), cmap=cmap)
    axes[2].set_title("ground truth")

    if target_tensor is not None:
        target_tensor = target_tensor.detach().cpu()
        # Plot target_tensor
        axes[3].imshow(target_tensor.squeeze(), cmap=cmap)
        axes[3].set_title("target")

    # Display the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"visualization_frame{ind}.png")


def calculate_mean_iou(preds, labels, smooth=1e-6):
    preds = torch.argmax(preds, dim=1)
    num_classes = preds.shape[1]
    labels = labels.squeeze(1)
    print(f"num_classes: {num_classes}")
    mean_iou = 0.0
    for cls in range(1, num_classes): 
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds[target_inds]).long().sum().item()
        total = (pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection)
        iou = (intersection + smooth) / (total + smooth)
        mean_iou += iou

    return mean_iou / num_classes

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

    focal_heights = (
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
    )

    model = AosDeepLab(len(focal_heights), 2)
    # model = UNetSmall(len(focal_heights), 2, pixel_out=False)
    print(f"GPU available: {check_gpu_availability()}")

    iteration, epoch = get_checkpoint(model, model.optimizer)

    trained_model = train_deeplab(model, focal_heights,  num_epochs=50, current_epoch=epoch, current_index=iteration)

    # torch.save(trained_model.state_dict(), "aosdeeplab_model.pth")
