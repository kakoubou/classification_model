# utils/evaluate.py

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

def test_model(model, test_dataset_path, transform, batch_size, save_dir="logs"):
    """
    Evaluate the model on a test dataset and generate evaluation metrics and plots.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_dataset_path (str): Path to the test dataset (organized in ImageFolder format).
        transform (callable): Transformation to apply to test images.
        batch_size (int): Batch size for DataLoader.
        save_dir (str): Directory to save the confusion matrix image and accuracy file.

    Returns:
        acc (float): Classification accuracy.
        cm (ndarray): Confusion matrix.
    """
    model.eval()
    model.cpu()

    all_preds = []
    all_labels = []

    test_dataset = ImageFolder(root=test_dataset_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Compute evaluation metrics
    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)

    # Visualize confusion matrix
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix\nAccuracy: {acc:.2%}')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    # Save accuracy to a file
    with open(os.path.join(save_dir, 'test_accuracy.txt'), 'w') as f:
        f.write(f'Accuracy: {acc:.4f} ({acc:.2%})')

    return acc, cm

def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

