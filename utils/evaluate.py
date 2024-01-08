import os
import numpy as np
import torch
from matplotlib import pyplot as plt

from checkpoint import load_checkpoint

# Example usage
# model = ...  # Your PyTorch model
# data_loader = ...  # Your PyTorch DataLoader for test data
# optimizer = ...  # Your PyTorch optimizer
# checkpoint_dir = ...  # Path to the directory containing checkpoints
def evaluation(checkpoint_dir, model, data_loader, criterion, optimizer=None):
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")])
    test_losses = []
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        model, optimiser = load_checkpoint(checkpoint_path, model, optimizer)
        loss, accuracy, specificity, precision, false_discovery_rate, recall, f1 = evaluate_model(model, data_loader, criterion)
        test_losses.append(loss)

    plt.plot(test_losses, marker='o', linestyle='-', color='b')
    plt.title('Test Loss Over Time')
    plt.xlabel('Test Iteration')
    plt.ylabel('Test Loss')
    plt.grid(True)
    plt.show()


# Example usage
# model = ...  # Your PyTorch model
# test_loader = ...  # Your PyTorch DataLoader for test data
# criterion = nn.CrossEntropyLoss()  # Use appropriate loss function for your task
# evaluate_model(model, test_loader, criterion)
def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    with torch.no_grad():
        for (inputs, labels) in data_loader:
            outputs = model(inputs)
            labels = labels.squeeze(dim=1)

            # check if outputs are dictionary
            if isinstance(outputs, dict):
                outputs = outputs["out"]

            loss = criterion(outputs, labels.long())
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            TP, FP, TN, FN = calculate_confusion_matrix(outputs, labels)

            true_positives += TP
            false_positives += FP
            true_negatives += TN
            false_negatives += FN
            total_samples += labels.size(0)

    # Calculate metrics
    accuracy = calculate_rates(true_positives + true_negatives, true_positives + true_negatives + false_positives + false_negatives)
    specificity = calculate_rates(true_negatives, true_negatives + false_positives)
    precision = calculate_rates(true_positives, true_positives + false_positives)
    false_discovery_rate = calculate_rates(false_positives, true_positives + false_positives)
    recall = calculate_rates(true_positives, true_positives + false_negatives)
    f1 = f1_score(precision, recall)
    test_loss = calculate_rates(total_loss, len(data_loader))

    print(f'Test Loss: {test_loss}')
    # print(f'Test Accuracy: {accuracy}')
    # print(f'Specificity: {specificity}')
    # print(f'Precision: {precision}')
    # print(f'False Discovery Rate: {false_discovery_rate}')
    # print(f'Recall: {recall}')
    # print(f'F1 Score: {f1}')
    # print()

    return (test_loss, accuracy, specificity, precision, false_discovery_rate, recall, f1)

def calculate_rates(numerator, denominator):
    if denominator == 0:
        return 0
    return numerator / denominator

def f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def calculate_confusion_matrix(outputs, labels):
    """
    Calculate confusion matrix values (TP, FP, TN, FN).

    Args:
        outputs (torch.Tensor): Model predictions.
        labels (torch.Tensor): True labels.

    Returns:
        int: True Positives (TP).
        int: False Positives (FP).
        int: True Negatives (TN).
        int: False Negatives (FN).
    """
    # Assuming binary classification, adjust for multi-class if needed
    predicted_classes = torch.argmax(outputs, dim=1)

    # Convert to numpy arrays for easier comparison
    predicted_np = predicted_classes.cpu().numpy()
    labels_np = labels.long().cpu().numpy()

    # Calculate TP, FP, TN, FN
    TP = np.sum((predicted_np == 1) & (labels_np == 1))
    FP = np.sum((predicted_np == 1) & (labels_np == 0))
    TN = np.sum((predicted_np == 0) & (labels_np == 0))
    FN = np.sum((predicted_np == 0) & (labels_np == 1))

    return TP, FP, TN, FN
