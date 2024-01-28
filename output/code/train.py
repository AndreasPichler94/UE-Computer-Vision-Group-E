import sys

import numpy as np
import torch
from PIL import Image
import os
from piq import ssim, psnr
import collections
from torchvision import transforms


from utils.aos_loader import _get_dataloader

# from evaluate import evaluate_model
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


def train_deeplab(
    model, focal_heights, num_epochs=10, current_index=0, current_epoch=0
):
    writer = SummaryWriter(f"trainlogs/deeplab_training_{num_epochs}_epochs")
    device = torch.device("cuda" if check_gpu_availability() else "cpu")
    res = (512, 512)
    batch_size = 10
    train_loader, _data = _get_dataloader(
        "train/",
        focal_heights=focal_heights,
        image_resolution=res,
        batch_size=batch_size,
    )
    test_loader, _ = _get_dataloader(
        "test/",
        focal_heights=focal_heights,
        image_resolution=res,
        batch_size=batch_size,
    )

    model.to(device)
    for state in model.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    rounding_threshold = 0.8  # round to 0 below this value
    deq = collections.deque(maxlen=50)  # last 50 loss values

    def get_running_loss():
        if len(deq):
            return np.mean(list(deq))
        else:
            return 0

    for epoch in range(current_epoch, num_epochs):
        _data.update_epoch(epoch)
        model.train()
        print(f"Number of samples {len(train_loader)}")
        print(f"Training epoch {epoch}")
        total_ssim = 0.0
        total_psnr = 0.0
        num_samples = 0

        for ind, (inputs, labels) in tqdm(enumerate(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)

            if model.model_name == "Deeplab" and model.pixel_out is False:
                threshold = 0.7843
                labels = torch.where(
                    labels > threshold,
                    torch.tensor(1.0, device=labels.device, dtype=torch.long),
                    torch.tensor(0.0, device=labels.device, dtype=torch.long),
                )

            model.optimizer.zero_grad()

            outputs = model(inputs)

            if model.model_name == "UNet":
                if model.pixel_out:
                    loss = model.criterion(outputs, labels)
                    prediction_tensor = outputs[0]
                    target_tensor = None
                else:
                    rounded = (labels > rounding_threshold).squeeze(1).long()
                    loss = model.criterion(outputs, rounded)
                    prediction_tensor = torch.softmax(outputs[0], 0)[1, :, :]
                    target_tensor = rounded[0]

                if (ind % 10) + 1 == 0:
                    print(f"Showing network outputs... Loss: {get_running_loss()}")
                    visualize_tensors(
                        "visualization",
                        epoch,
                        ind,
                        input_tensor=inputs[0][0],
                        prediction_tensor=prediction_tensor,
                        target_tensor=target_tensor,
                        ground_truth=labels[0],
                    )
            elif model.model_name == "Deeplab":
                if ind % 10 == 0:
                    print("Showing network outputs...")
                    if model.pixel_out:
                        visualize_tensors(
                            "visualization",
                            epoch,
                            ind,
                            input_tensor=inputs[0][0],
                            prediction_tensor=outputs["out"][0],
                            target_tensor=labels[0],
                            ground_truth=labels[0],
                        )
                    else:
                        visualize_tensors(
                            "visualization",
                            epoch,
                            ind,
                            input_tensor=inputs[0][0],
                            prediction_tensor=torch.argmax(
                                outputs["out"][0], dim=0, keepdim=True
                            ),
                            target_tensor=labels[0],
                            ground_truth=labels[0],
                        )
                        visualize_tensors(
                            "visualization_probabilities",
                            epoch,
                            ind,
                            input_tensor=inputs[0][0],
                            prediction_tensor=torch.softmax(outputs["out"][0], 0)[
                                1, :, :
                            ],
                            target_tensor=labels[0],
                            ground_truth=labels[0],
                        )

                if model.model_name == "Deeplab":
                    if model.pixel_out:
                        loss = model.criterion(outputs["out"], labels)
                        normalized_outputs = torch.sigmoid(outputs["out"])
                        total_ssim += ssim(normalized_outputs, labels).item()
                        total_psnr += psnr(normalized_outputs, labels).item()
                    else:
                        loss = model.criterion(outputs["out"], labels.squeeze(1).long())
                else:
                    loss = model.criterion(outputs["out"], labels.squeeze(1).long())

            else:
                raise NotImplementedError("Only UNet and Deeplab implemented")

            num_samples += 1
            loss.backward()
            model.optimizer.step()

            deq.append(loss.item())

            if (ind + 1) % 1000 == 0:
                print(f"Iteration {ind}, Loss: {get_running_loss()}")
                save_checkpoint(
                    model, model.optimizer, ind, epoch, checkpoint_dir="./checkpoints"
                )
                avg_ssim = total_ssim / num_samples
                avg_psnr = total_psnr / num_samples
                writer.add_scalar(
                    "SSIM/train", avg_ssim, epoch * len(train_loader) + ind
                )
                writer.add_scalar(
                    "PSNR/train", avg_psnr, epoch * len(train_loader) + ind
                )
                writer.add_scalar(
                    "Loss/train", get_running_loss(), epoch * len(train_loader) + ind
                )
                total_ssim = 0.0
                total_psnr = 0.0
                num_samples = 0

        print(f"Epoch {epoch}, Loss: {get_running_loss()}")
        if (epoch + 1) % 50 == 0:
            save_checkpoint(
                        model, model.optimizer, 0, epoch, checkpoint_dir="./checkpoints"
                    )

        model.eval()
        valid_loss = 0.0
        total_iou = 0.0
        total_batches = 0
        total_ssim_valid = 0.0
        total_psnr_valid = 0.0
        num_samples_valid = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                if model.model_name == "UNet":
                    if model.pixel_out:
                        prediction_tensor = outputs[0]
                        target_tensor = None
                        loss = model.criterion(outputs, labels)
                        total_ssim_valid += ssim(outputs, labels).item()
                        total_psnr_valid += psnr(outputs, labels).item()

                        print(f"Showing network outputs... Loss: {get_running_loss()}")
                        visualize_tensors(
                            "visualization_valid",
                            epoch,
                            0,
                            input_tensor=inputs[0][0],
                            prediction_tensor=prediction_tensor,
                            target_tensor=target_tensor,
                            ground_truth=labels[0],
                        )
                    else:
                        rounded = (labels > rounding_threshold).squeeze(1).long()
                        loss = model.criterion(outputs, rounded)
                else:
                    if model.pixel_out:
                        loss = model.criterion(outputs["out"], labels)
                        normalized_outputs = torch.sigmoid(outputs["out"])
                        total_ssim_valid += ssim(normalized_outputs, labels).item()
                        total_psnr_valid += psnr(normalized_outputs, labels).item()
                        if model.model_name == "Deeplab":
                            print("Showing network outputs...")
                            visualize_tensors(
                                "visualization_valid",
                                epoch,
                                num_samples_valid,
                                input_tensor=inputs[0][0],
                                prediction_tensor=outputs["out"][0],
                                target_tensor=labels[0],
                                ground_truth=labels[0],
                            )
                    else:
                        loss = model.criterion(outputs["out"], labels.squeeze(1).long())

                num_samples_valid += 1
                valid_loss += loss.item()

                preds = outputs["out"] if model.model_name != "UNet" else outputs

                if model.model_name == "Deeplab" and model.pixel_out is False:
                    mean_iou = calculate_mean_iou(preds, labels.squeeze(1))
                    total_iou += mean_iou
                total_batches += 1

            avg_valid_loss = valid_loss / total_batches
            avg_valid_iou = total_iou / total_batches
            avg_ssim_valid = total_ssim_valid / num_samples_valid
            avg_psnr_valid = total_psnr_valid / num_samples_valid
            writer.add_scalar("SSIM/validation", avg_ssim_valid, epoch)
            writer.add_scalar("PSNR/validation", avg_psnr_valid, epoch)
            print(f"Validation Loss: {avg_valid_loss}")
            print(f"Validation Mean IoU: {avg_valid_iou}")
            writer.add_scalar("Loss/validation", avg_valid_loss, epoch)
            writer.add_scalar("Mean IoU/validation", avg_valid_iou, epoch)

    # evaluate_model(model, train_loader, torch.nn.CrossEntropyLoss())
    writer.close()


def visualize_tensors(
    filename,
    epoch,
    ind,
    input_tensor,
    prediction_tensor,
    target_tensor,
    ground_truth,
    cmap="gray",
):
    # Ensure the tensors are detached and moved to cpu
    input_tensor = input_tensor.detach().cpu()
    prediction_tensor = prediction_tensor.detach().cpu()
    ground_truth = ground_truth.detach().cpu()

    # Create a subplot with 3 columns for the 3 images
    fig, axes = plt.subplots(
        nrows=1,
        ncols=(4 if target_tensor is not None else 3),
        figsize=(15 + (5 if target_tensor is not None else 0), 5),
    )

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
    plt.savefig(f"{filename}_{epoch}_{ind}.png")
    plt.close("all")


def calculate_mean_iou(preds, labels, smooth=1e-6):
    preds = torch.argmax(preds, dim=1)
    preds = preds.view(-1)
    labels = labels.view(-1).long()

    num_classes = max(preds.max(), labels.max()) + 1
    mean_iou = 0.0
    for cls in range(1, num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item() - intersection

        iou = (intersection + smooth) / (union + smooth)
        mean_iou += iou

    return mean_iou / (num_classes - 1)


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

    model = AosDeepLab(len(focal_heights), 1, pixel_out=True)
    print(f"GPU available: {check_gpu_availability()}")

    iteration, epoch = get_checkpoint(model, model.optimizer, check_gpu_availability())

    trained_model = train_deeplab(
        model,
        focal_heights,
        num_epochs=20,
        current_epoch=epoch,
        current_index=iteration,
    )

    torch.save(trained_model.state_dict(), "deeplab_resnet101_20_epochs.pth")