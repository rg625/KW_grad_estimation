import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
import numpy as np

# Check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters
BATCH_SIZE = 1024  # Reduced batch size

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.MNIST(root='~/datasets', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Simple CNN Model for MNIST
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(model_path):
    model = CNNModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def plot_confusion_matrix_cbar(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.savefig(f'optimizer_plots/mnist_classifier/confusion_matrix_{model_name}.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.savefig(f'optimizer_plots/mnist_classifier/confusion_matrix_{model_name}.png')
    plt.close()

def create_prediction_figures(models, test_loader, device):
    for name, model in models.items():
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        with torch.no_grad():
            images, labels = next(iter(test_loader))
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            sampled_indices = np.random.choice(len(labels), 16, replace=False)
            for i, idx in enumerate(sampled_indices):
                ax = axes[i // 4, i % 4]
                ax.imshow(images[idx].cpu().squeeze(), cmap='gray')
                ax.set_title(f'T:{labels[idx].item()} P:{predicted[idx].item()}')
                ax.axis('off')
        fig.suptitle(f'Predictions by {name} Optimizer')
        plt.tight_layout()
        plt.savefig(f'optimizer_plots/mnist_classifier/predictions_{name}.png')
        plt.close()

        # Generate confusion matrix
        y_true = labels.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        plot_confusion_matrix(y_true, y_pred, name)

def main():
    model_dir = Path('optimizer_models')
    models = {
        'SPSAOptimizerSimple': load_model(model_dir / 'SPSAOptimizerSimple_model.pth'),
        'SPSAOptimizerAdaptive': load_model(model_dir / 'SPSAOptimizerAdaptive_model.pth'),
        'KieferWolfowitzSimple': load_model(model_dir / 'KieferWolfowitzOptimizerSimple_model.pth'),
        'KieferWolfowitzAdaptive': load_model(model_dir / 'KieferWolfowitzOptimizerAdaptive_model.pth'),
        'Adam': load_model(model_dir / 'Adam_model.pth'),
        'Adagrad': load_model(model_dir / 'Adagrad_model.pth'),
        'SGD': load_model(model_dir / 'SGD_model.pth'),
    }
    
    create_prediction_figures(models, test_loader, DEVICE)



if __name__ == '__main__':
    main()
