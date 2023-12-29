import torch
import torch.nn as nn

# Example usage
# model = ...  # Your PyTorch model
# test_loader = ...  # Your PyTorch DataLoader for test data
# criterion = nn.CrossEntropyLoss()  # Use appropriate loss function for your task
# evaluate_model(model, test_loader, criterion)
def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for (inputs, labels) in test_loader:
            #print("inputs", inputs.shape)
            #print("labels", labels.shape)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    # Calculate metrics
    accuracy = accuracy_score(correct_predictions, total_samples)
    precision = precision_score(correct_predictions, total_samples)
    recall = recall_score(correct_predictions, total_samples)
    f1 = f1_score(correct_predictions, total_samples)

    print(f'Test Loss: {total_loss / len(test_loader)}')
    print(f'Test Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

def accuracy_score(correct_predictions, total_samples):
    return correct_predictions / total_samples

def precision_score(correct_predictions, total_samples):
    return correct_predictions / total_samples

def recall_score(correct_predictions, total_samples):
    return correct_predictions / total_samples

def f1_score(correct_predictions, total_samples):
    precision = precision_score(correct_predictions, total_samples)
    recall = recall_score(correct_predictions, total_samples)
    return 2 * precision * recall / (precision + recall)
