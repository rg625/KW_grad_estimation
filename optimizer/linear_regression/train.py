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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pathlib import Path
import seaborn as sns
import time
import tracemalloc

# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters
BATCH_SIZE = 100
N_Datapoints = 10000
N_EPOCHS = 200
N_Features = 20
LEARNING_RATE = 0.1

# True parameters for synthetic dataset
TRUE_WEIGHTS = np.random.rand(N_Features)
TRUE_BIAS = 2.0

# Synthetic dataset for linear regression with 10 features
X = np.random.rand(N_Datapoints, N_Features)
y = X @ TRUE_WEIGHTS + TRUE_BIAS + 0.1 * np.random.randn(N_Datapoints)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

# Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, N_Features):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(N_Features, 1)

    def forward(self, x):
        return self.linear(x)

# Training function
def train_model(optimizer_class, optimizer_params, device, num_epochs=N_EPOCHS):
    model = LinearRegressionModel(N_Features).to(device)
    criterion = nn.MSELoss()
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
        
        progress_bar = tqdm(range(0, len(X_train), BATCH_SIZE), desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=False)
        for i in progress_bar:
            batch_X = X_train[i:i+BATCH_SIZE]
            batch_y = y_train[i:i+BATCH_SIZE]
            
            def closure():
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                return loss

            start_time = time.time()
            loss = optimizer.step(closure)
            end_time = time.time()
            
            batch_time = end_time - start_time
            running_loss += loss.item() * batch_X.size(0)
            times.append(batch_time)
            
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))
        
        avg_loss = running_loss / len(X_train)
        train_losses.append(avg_loss)
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        times.append(epoch_time)
        
        # Evaluate on training set
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train).squeeze()
            train_accuracy = r2_score(y_train.cpu().numpy(), train_outputs.cpu().numpy()) * 100
        train_accuracies.append(train_accuracy)
        
        # Evaluate on test set
        with torch.no_grad():
            test_outputs = model(X_test).squeeze()
            test_loss = criterion(test_outputs, y_test).item()
            test_accuracy = r2_score(y_test.cpu().numpy(), test_outputs.cpu().numpy()) * 100
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # Record parameter values
        for name, param in model.named_parameters():
            param_hist[name].append(param.data.cpu().numpy().copy())
        
        # Measure memory usage
        current, peak = tracemalloc.get_traced_memory()
        compute_costs.append(peak / 10**6)  # Convert to MB
        tracemalloc.stop()
        
        # Measure complexity (O(n))
        complexity = len(X_train) * len(list(model.parameters()))
        complexities.append(complexity)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%, Compute Cost: {compute_costs[-1]:.2f} MB, Complexity: O({len(X_train)}), Time: {epoch_time:.4f} seconds')

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

def plot_param_changes(plots_dir, all_param_hist, true_params):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    for param_name in true_params:
        true_values = true_params[param_name]
        for i in range(len(true_values)):
            plt.figure(figsize=(16, 9))
            for optimizer_name, param_hist in all_param_hist.items():
                values = np.array(param_hist[param_name])
                if param_name == 'linear.bias':
                    plt.plot(values, label=f'{optimizer_name} {param_name}')
                else:
                    plt.plot(values.squeeze()[:, i], label=f'{optimizer_name} {param_name}_{i}')
            if param_name == 'linear.bias':
                plt.axhline(y=true_values[0], color='r', linestyle='--', label=f'True {param_name}')
            else:
                plt.axhline(y=true_values[i], color='r', linestyle='--', label=f'True {param_name}_{i}')
            plt.xlabel('Epoch')
            plt.ylabel('Parameter Value')
            plt.title(f'{param_name}_{i} changes for all optimizers')
            plt.legend()
            plt.savefig(plots_dir / f'{param_name}_{i}_changes.png')
            plt.clf()
            plt.close()

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

    all_train_losses = {}
    all_test_losses = {}
    all_train_accuracies = {}
    all_test_accuracies = {}
    all_compute_costs = {}
    all_complexities = {}
    all_times = {}
    all_param_hist = {}
    true_params = {'linear.weight': TRUE_WEIGHTS, 'linear.bias': np.array([TRUE_BIAS])}
    
    for name, (optimizer_class, optimizer_params) in optimizers.items():
        print(f'Training with {name} optimizer...')
        train_losses, test_losses, train_accuracies, test_accuracies, param_hist, compute_costs, complexities, times, model = train_model(optimizer_class, optimizer_params, device, num_epochs)
        all_train_losses[name] = {'train_losses': train_losses}
        all_test_losses[name] = {'test_losses': test_losses}
        all_train_accuracies[name] = {'train_accuracies': train_accuracies}
        all_test_accuracies[name] = {'test_accuracies': test_accuracies}
        all_compute_costs[name] = {'compute_costs': compute_costs}
        all_complexities[name] = {'complexities': complexities}
        all_times[name] = {'times': times}
        all_param_hist[name] = param_hist

        # Create directory for the optimizer
        optimizer_dir = Path(f'optimizer_plots/linear_regression/linear_regression/{name}')
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
    # Plot parameter changes for all optimizers
    plot_param_changes('optimizer_plots/linear_regression/parameters', all_param_hist, true_params)

    # Plot comparative results for training losses
    comparative_dir = Path('optimizer_plots/linear_regression/comparative')
    comparative_dir.mkdir(parents=True, exist_ok=True)
    
    for name, results in all_train_losses.items():
        plt.plot(results['train_losses'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss vs Epoch for Different Optimizers')
    plt.legend()
    plt.savefig(comparative_dir / 'comparative_train_loss.png')
    plt.clf()
    plt.close()
    
    # Plot comparative results for test losses
    for name, results in all_test_losses.items():
        plt.plot(results['test_losses'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss vs Epoch for Different Optimizers')
    plt.legend()
    plt.savefig(comparative_dir / 'comparative_test_loss.png')
    plt.clf()
    plt.close()

    # Plot comparative results for training accuracies
    for name, results in all_train_accuracies.items():
        plt.plot(results['train_accuracies'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train Accuracy vs Epoch for Different Optimizers')
    plt.legend()
    plt.savefig(comparative_dir / 'comparative_train_accuracy.png')
    plt.clf()
    plt.close()
    # Plot comparative results for test accuracies
    for name, results in all_test_accuracies.items():
        plt.plot(results['test_accuracies'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy vs Epoch for Different Optimizers')
    plt.legend()
    plt.savefig(comparative_dir / 'comparative_test_accuracy.png')
    plt.clf()
    plt.close()
    
    # Plot comparative results for compute costs
    for name, results in all_compute_costs.items():
        plt.plot(results['compute_costs'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Compute Cost vs Epoch for Different Optimizers')
    plt.legend()
    plt.savefig(comparative_dir / 'comparative_compute_cost.png')
    plt.clf()
    plt.close()
    # Plot comparative results for complexities
    for name, results in all_complexities.items():
        plt.plot(results['complexities'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Complexity (O(n))')
    plt.title('Complexity vs Epoch for Different Optimizers')
    plt.legend()
    plt.savefig(comparative_dir / 'comparative_complexity.png')
    plt.clf()
    plt.close()
    
    # Plot comparative results for times
    for name, results in all_times.items():
        plt.plot(results['times'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Time vs Epoch for Different Optimizers')
    plt.legend()
    plt.savefig(comparative_dir / 'comparative_time.png')
    plt.clf()
    plt.close()

if __name__ == '__main__':
    main()