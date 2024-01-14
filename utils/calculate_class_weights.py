import torch
from aos_loader import _get_dataloader


def calculate_class_frequency(dataloader):
    count_dict = {0: 0, 1: 0}

    temp_idx = 0
    for _, labels in dataloader:
        labels = labels.long()
        unique, counts = torch.unique(labels, return_counts=True)
        for i, c in zip(unique, counts):
            count_dict[int(i)] += c.item()
        # if temp_idx > 500:
        #     break
        # temp_idx += 1

    total = sum(count_dict.values())

    frequency_dict = {k: v / total for k, v in count_dict.items()}

    return frequency_dict


def calculate_class_weights(frequency_dict):
    weights = {k: 1 / v if v > 0 else 0 for k, v in frequency_dict.items()}

    total = sum(weights.values())
    normalized_weights = {k: (v * len(weights)) / total for k, v in weights.items()}

    return normalized_weights

res = (512, 512)
batch_size = 15
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

train_loader = _get_dataloader(
        "./data/train/",
        focal_heights=focal_heights,
        image_resolution=res,
        batch_size=batch_size,
    )

class_frequencies = calculate_class_frequency(train_loader)
print("Class Frequencies: ", class_frequencies)

class_weights = calculate_class_weights(class_frequencies)
print("Class Weights: ", class_weights)
