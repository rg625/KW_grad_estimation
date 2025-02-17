import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from kifwolfoptimizer.optimizer_simple import KieferWolfowitzOptimizerSimple
from kifwolfoptimizer.optimizer_adaptive import KieferWolfowitzOptimizerAdaptive
from spsa.optimizer_simple import SPSAOptimizerSimple
from spsa.optimizer_adaptive import SPSAOptimizerAdaptive
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from pathlib import Path
import seaborn as sns
import time
import tracemalloc
import gc
import json

# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters
BATCH_SIZE = 4096  # Reduced batch size
N_EPOCHS = 10
LEARNING_RATE = 0.01

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='~/datasets', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='~/datasets', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
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

# Training function
def train_model(optimizer_class, optimizer_params, device, num_epochs=N_EPOCHS):
    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    param_hist = {name: [] for name, _ in model.named_parameters()}
    compute_costs = []
    complexities = []
    times = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        tracemalloc.start()
        
        progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=False)
        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            def closure():
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                return loss

            start_time = time.time()
            try:
                loss = optimizer.step(closure)
            except torch.cuda.OutOfMemoryError:
                print("CUDA out of memory. Skipping batch.")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            end_time = time.time()
            
            batch_time = end_time - start_time
            running_loss += loss.item() * images.size(0)
            times.append(batch_time)
            
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))
        
        avg_loss = running_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        times.append(epoch_time)
        
        # Evaluate on training set
        model.eval()
        with torch.no_grad():
            train_outputs = []
            train_labels = []
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                train_outputs.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            train_accuracy = accuracy_score(train_labels, train_outputs) * 100
        train_accuracies.append(train_accuracy)
        
        # Evaluate on test set
        with torch.no_grad():
            test_outputs = []
            test_labels = []
            running_test_loss = 0.0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_outputs.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                test_loss = criterion(outputs, labels)
                running_test_loss += test_loss.item() * images.size(0)
            test_accuracy = accuracy_score(test_labels, test_outputs) * 100
            test_losses.append(running_test_loss / len(test_loader.dataset))
        test_accuracies.append(test_accuracy)
        
        # Record parameter values
        for name, param in model.named_parameters():
            param_hist[name].append(param.data.cpu().numpy().copy())
        
        # Measure memory usage
        current, peak = tracemalloc.get_traced_memory()
        compute_costs.append(peak / 10**6)  # Convert to MB
        tracemalloc.stop()
        
        # Measure complexity (O(n))
        complexity = len(train_loader.dataset) * len(list(model.parameters()))
        complexities.append(complexity)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Test Loss: {test_losses[-1]:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%, Compute Cost: {compute_costs[-1]:.2f} MB, Complexity: O({len(train_loader.dataset)}), Time: {epoch_time:.4f} seconds')

    # Save the model after training
    model_dir = Path('optimizer_models')
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / f'{optimizer_class.__name__}_model.pth')

    return train_losses, test_losses, train_accuracies, test_accuracies, param_hist, compute_costs, complexities, times, model

# Plot results
def plot_results(plots_dir, runs, result_name):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    imgpath = plots_dir / f'{result_name}.png'
    if imgpath.exists():
        imgpath.unlink()

    if not imgpath.is_file():
        try:
            plt.figure(figsize=(16, 9))
            c_cycler = sns.color_palette()
            for name, results in runs.items():
                if result_name in results and len(results[result_name]):
                    plt.plot(range(1, len(results[result_name]) + 1),
                             results[result_name],
                             '.-', label=name)
            plt.xlabel('Epoch')
            if 'accuracy' in result_name:
                plt.ylabel('Accuracy (%)')
            elif 'cost' in result_name:
                plt.ylabel('Memory Usage (MB)')
            elif 'complexity' in result_name:
                plt.ylabel('Complexity (O(n))')
            else:
                plt.ylabel(result_name.split('_')[1].capitalize())
            plt.title(result_name.replace('_', ' ').capitalize())
            plt.legend()
            plt.savefig(imgpath)
            plt.close()
        except Exception as e:
            print(f"Failed to plot {result_name}: {e}")

def plot_param_changes(plots_dir, all_param_hist):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    for param_name in all_param_hist[list(all_param_hist.keys())[0]]:
        for i in range(len(all_param_hist[list(all_param_hist.keys())[0]][param_name][0])):
            plt.figure(figsize=(16, 9))
            for optimizer_name, param_hist in all_param_hist.items():
                values = np.array(param_hist[param_name])
                plt.plot(values[:, i], label=f'{optimizer_name} {param_name}_{i}')
            plt.xlabel('Epoch')
            plt.ylabel('Parameter Value')
            plt.title(f'{param_name}_{i} changes for all optimizers')
            plt.legend()
            plt.savefig(plots_dir / f'{param_name}_{i}_changes.png')
            plt.clf()
            plt.close()

def save_models(models, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        torch.save(model.state_dict(), directory / f'{name}_model.pth')

def create_prediction_figure(models, test_loader, device):
    plt.figure(figsize=(16, 9))
    for idx, (name, model) in enumerate(models.items()):
        model.eval()
        with torch.no_grad():
            images, labels = next(iter(test_loader))
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(10):
                plt.subplot(len(models), 10, idx * 10 + i + 1)
                plt.imshow(images[i].cpu().squeeze(), cmap='gray')
                plt.title(f'{predicted[i].item()}')
                plt.axis('off')
        plt.suptitle(f'Predictions by {name} Optimizer')
    plt.savefig('optimizer_plots/mnist_classifier/predictions.png')
    plt.close()

def save_results_as_json(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = N_EPOCHS
    optimizers = {
        'SPSAOptimizerSimple': (SPSAOptimizerSimple, {'lr': LEARNING_RATE, 'perturbation': 0.05}),
        'SPSAOptimizerAdaptive': (SPSAOptimizerAdaptive, {'lr': LEARNING_RATE, 'perturbation': 0.05}),
        'KieferWolfowitzSimple': (KieferWolfowitzOptimizerSimple, {'lr': LEARNING_RATE, 'perturbation': 0.05}),
        'KieferWolfowitzAdaptive': (KieferWolfowitzOptimizerAdaptive, {'lr': LEARNING_RATE, 'perturbation': 0.05}),
        'Adam': (optim.Adam, {'lr': LEARNING_RATE}),
        'Adagrad': (optim.Adagrad, {'lr': LEARNING_RATE}),
        'SGD': (optim.SGD, {'lr': LEARNING_RATE}),
    }

    all_results = {
        'train_losses': {},
        'test_losses': {},
        'train_accuracies': {},
        'test_accuracies': {},
        'compute_costs': {},
        'complexities': {},
        'times': {},
        'param_hist': {},
    }
    models = {}
    
    for name, (optimizer_class, optimizer_params) in optimizers.items():
        print(f'Training with {name} optimizer...')
        train_losses, test_losses, train_accuracies, test_accuracies, param_hist, compute_costs, complexities, times, model = train_model(optimizer_class, optimizer_params, device, num_epochs)
        all_results['train_losses'][name] = train_losses
        all_results['test_losses'][name] = test_losses
        all_results['train_accuracies'][name] = train_accuracies
        all_results['test_accuracies'][name] = test_accuracies
        all_results['compute_costs'][name] = compute_costs
        all_results['complexities'][name] = complexities
        all_results['times'][name] = times
        all_results['param_hist'][name] = param_hist
        models[name] = model

        # Create directory for the optimizer
        optimizer_dir = Path(f'optimizer_plots/mnist_classifier/{name}')
        optimizer_dir.mkdir(parents=True, exist_ok=True)

        # Plot individual loss
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss vs Epoch for {name} Optimizer')
        plt.legend()
        plt.savefig(optimizer_dir / f'{name}_optimizer_loss.png')
        plt.clf()

        # Plot individual accuracy
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Accuracy vs Epoch for {name} Optimizer')
        plt.legend()
        plt.savefig(optimizer_dir / f'{name}_optimizer_accuracy.png')
        plt.clf()

        # Plot individual compute cost
        plt.plot(compute_costs, label='Compute Cost')
        plt.xlabel('Epoch')
        plt.ylabel('Memory Usage (MB)')
        plt.title(f'Compute Cost vs Epoch for {name} Optimizer')
        plt.legend()
        plt.savefig(optimizer_dir / f'{name}_optimizer_compute_cost.png')
        plt.clf()

        # Plot individual complexity
        plt.plot(complexities, label='Complexity')
        plt.xlabel('Epoch')
        plt.ylabel('Complexity (O(n))')
        plt.title(f'Complexity vs Epoch for {name} Optimizer')
        plt.legend()
        plt.savefig(optimizer_dir / f'{name}_optimizer_complexity.png')
        plt.clf()
        plt.close()

        # Plot individual time
        plt.plot(times, label='Time')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.title(f'Time vs Epoch for {name} Optimizer')
        plt.legend()
        plt.savefig(optimizer_dir / f'{name}_optimizer_time.png')
        plt.clf()
        plt.close()
    
    # Save results as JSON
    save_results_as_json(all_results, 'optimizer_results.json')

    # Plot parameter changes for all optimizers
    plot_param_changes('optimizer_plots/mnist_classifier/parameters', all_results['param_hist'])

    # Plot comparative results for training losses
    comparative_dir = Path('optimizer_plots/mnist_classifier/comparative')
    comparative_dir.mkdir(parents=True, exist_ok=True)
    
    for name, results in all_results['train_losses'].items():
        plt.plot(results, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss vs Epoch for Different Optimizers')
    plt.legend()
    plt.savefig(comparative_dir / 'comparative_train_loss.png')
    plt.clf()
    plt.close()
    
    # Plot comparative results for test losses
    for name, results in all_results['test_losses'].items():
        plt.plot(results, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss vs Epoch for Different Optimizers')
    plt.legend()
    plt.savefig(comparative_dir / 'comparative_test_loss.png')
    plt.clf()
    plt.close()

    # Plot comparative results for training accuracies
    for name, results in all_results['train_accuracies'].items():
        plt.plot(results, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train Accuracy vs Epoch for Different Optimizers')
    plt.legend()
    plt.savefig(comparative_dir / 'comparative_train_accuracy.png')
    plt.clf()
    plt.close()
    
    # Plot comparative results for test accuracies
    for name, results in all_results['test_accuracies'].items():
        plt.plot(results, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy vs Epoch for Different Optimizers')
    plt.legend()
    plt.savefig(comparative_dir / 'comparative_test_accuracy.png')
    plt.clf()
    plt.close()
    
    # Plot comparative results for compute costs
    for name, results in all_results['compute_costs'].items():
        plt.plot(results, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Compute Cost vs Epoch for Different Optimizers')
    plt.legend()
    plt.savefig(comparative_dir / 'comparative_compute_cost.png')
    plt.clf()
    plt.close()
    
    # Plot comparative results for complexities
    for name, results in all_results['complexities'].items():
        plt.plot(results, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Complexity (O(n))')
    plt.title('Complexity vs Epoch for Different Optimizers')
    plt.legend()
    plt.savefig(comparative_dir / 'comparative_complexity.png')
    plt.clf()
    plt.close()
    
    # Plot comparative results for times
    for name, results in all_results['times'].items():
        plt.plot(results, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Time vs Epoch for Different Optimizers')
    plt.legend()
    plt.savefig(comparative_dir / 'comparative_time.png')
    plt.clf()
    plt.close()

    # Save models
    save_models(models, 'optimizer_models')

    # Create prediction figure
    create_prediction_figure(models, test_loader, device)

if __name__ == '__main__':
    main()