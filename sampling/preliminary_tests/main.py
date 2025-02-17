import torch
import time
import psutil
import plotly.graph_objects as go
import plotly.express as px
import json
import os

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a 1D Gaussian Mixture Model (GMM)
def log_prob(x):
    """Compute log probability of a Gaussian mixture model."""
    x = x.to(device).requires_grad_(True)
    p1 = torch.exp(-0.5 * ((x - 2) / 0.8) ** 2) / (0.8 * (2 * torch.pi) ** 0.5)
    p2 = torch.exp(-0.5 * ((x + 2) / 0.8) ** 2) / (0.8 * (2 * torch.pi) ** 0.5)
    return torch.log(0.5 * p1 + 0.5 * p2 + 1e-9)  # Small constant for numerical stability

# Compute the true score function using autograd
def true_score(x, step, step_size):
    x = x.to(device)
    log_p = log_prob(x)
    grad = step_size * torch.autograd.grad(log_p.sum(), x, create_graph=True)[0]
    return grad.detach(), step_size

# Estimate the score function using finite differences
def KW(x, step, step_size, delta=5e-2):
    x = x.to(device)
    step_size = (1 + step_size) ** 0.6
    delta = step_size * torch.tensor(delta / (step + 1) ** 0.5, device=device)  # Adaptive step size
    return (log_prob(x + delta) - log_prob(x - delta)) / (2 * delta), step_size

# SPSA Gradient Estimation
def spsa_gradient(x, step, step_size, delta=5e-2):
    x = x.to(device)
    step_size = (1 + step_size) ** 0.6
    perturbation = torch.empty_like(x).uniform_(-1, 1).sign()
    delta = step_size * torch.tensor(delta / (step + 1) ** 0.5, device=device)  # Adaptive step size
    x_plus = x + delta * perturbation
    x_minus = x - delta * perturbation
    gradient_estimate = (log_prob(x_plus) - log_prob(x_minus)) / (2 * delta * perturbation)
    return gradient_estimate, step_size

# Measure memory usage on CUDA
def get_memory_usage():
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensure all GPU operations are completed
        return torch.cuda.max_memory_allocated() / (1024**2)  # Peak memory in MB
    else:
        return psutil.Process().memory_info().rss / (1024**2)  # CPU memory usage

# Langevin Dynamics sampler with better exploration
def langevin_dynamics(x0_list, score_function, steps=100, eta=0.5, noise_scale=0.1):
    """Perform Langevin Dynamics using a specified score function, with multiple initial points."""
    x = x0_list.to(device)
    samples = []

    for i in range(steps):
        score, step_size = score_function(x, i, eta)  # Compute score function
        noise = torch.randn_like(x, device=device)  # Reduce noise over time
        x = x + score + torch.sqrt(torch.tensor(2 * step_size, device=device)) * noise
        samples.append(x.clone().detach())

    return torch.cat(samples)

# Hamiltonian Monte Carlo sampler without Metropolis step
def hmc_sampler(x0_list, score_function, steps=100, leapfrog_steps=3, step_size=0.1):
    """Perform Hamiltonian Monte Carlo sampling with multiple initial points using a specified score function."""
    x = x0_list.to(device)
    samples = []

    for _ in range(steps):
        x = x.clone().detach().requires_grad_(True)
        p = torch.randn_like(x, device=device)

        for _ in range(leapfrog_steps):
            score, _ = score_function(x, _, step_size)
            p = p - 0.5 * step_size * score
            x = x + step_size * p
            score, _ = score_function(x, _, step_size)
            p = p - 0.5 * step_size * score
        
        samples.append(x.clone().detach())

    return torch.cat(samples)

# Experiment configurations
num_samples_list = [1, 10, 100, 1000, 10000]
steps_list = [1, 10, 100, 1000, 10000]
results = []

# Create results directory if it doesn't exist
results_dir = "results_univariate"
os.makedirs(results_dir, exist_ok=True)

# Run experiments
for num_samples in num_samples_list:
    for steps in steps_list:
        x0_list = torch.randn(num_samples, device=device)

        # Benchmarking True Gradients (Langevin)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # Reset memory tracking
        start_time = time.perf_counter()
        mem_before = get_memory_usage()
        samples_true = langevin_dynamics(x0_list, true_score, steps)
        torch.cuda.synchronize()
        mem_after = get_memory_usage()
        time_true = time.perf_counter() - start_time
        mem_true = mem_after - mem_before

        # Benchmarking Finite Differences (Langevin)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # Reset memory tracking
        start_time = time.perf_counter()
        mem_before = get_memory_usage()
        samples_estimated = langevin_dynamics(x0_list, KW, steps)
        torch.cuda.synchronize()
        mem_after = get_memory_usage()
        time_estimated = time.perf_counter() - start_time
        mem_estimated = mem_after - mem_before

        # Benchmarking SPSA (Langevin)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # Reset memory tracking
        start_time = time.perf_counter()
        mem_before = get_memory_usage()
        samples_spsa = langevin_dynamics(x0_list, spsa_gradient, steps)
        torch.cuda.synchronize()
        mem_after = get_memory_usage()
        time_spsa = time.perf_counter() - start_time
        mem_spsa = mem_after - mem_before

        # Benchmarking True Gradients (HMC)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # Reset memory tracking
        start_time = time.perf_counter()
        mem_before = get_memory_usage()
        samples_hmc_true = hmc_sampler(x0_list, true_score, steps)
        torch.cuda.synchronize()
        mem_after = get_memory_usage()
        time_hmc_true = time.perf_counter() - start_time
        mem_hmc_true = mem_after - mem_before

        # Benchmarking Finite Differences (HMC)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # Reset memory tracking
        start_time = time.perf_counter()
        mem_before = get_memory_usage()
        samples_hmc_estimated = hmc_sampler(x0_list, KW, steps)
        torch.cuda.synchronize()
        mem_after = get_memory_usage()
        time_hmc_estimated = time.perf_counter() - start_time
        mem_hmc_estimated = mem_after - mem_before

        # Benchmarking SPSA (HMC)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # Reset memory tracking
        start_time = time.perf_counter()
        mem_before = get_memory_usage()
        samples_hmc_spsa = hmc_sampler(x0_list, spsa_gradient, steps)
        torch.cuda.synchronize()
        mem_after = get_memory_usage()
        time_hmc_spsa = time.perf_counter() - start_time
        mem_hmc_spsa = mem_after - mem_before

        # Collect all steps for convergence analysis
        all_steps_true = langevin_dynamics(x0_list, true_score, steps).cpu().numpy()
        all_steps_estimated = langevin_dynamics(x0_list, KW, steps).cpu().numpy()
        all_steps_spsa = langevin_dynamics(x0_list, spsa_gradient, steps).cpu().numpy()
        all_steps_hmc_true = hmc_sampler(x0_list, true_score, steps).cpu().numpy()
        all_steps_hmc_estimated = hmc_sampler(x0_list, KW, steps).cpu().numpy()
        all_steps_hmc_spsa = hmc_sampler(x0_list, spsa_gradient, steps).cpu().numpy()

        # Collect results
        results.append({
            "num_samples": num_samples,
            "steps": steps,
            "time_true": time_true,
            "mem_true": mem_true,
            "time_estimated": time_estimated,
            "mem_estimated": mem_estimated,
            "time_spsa": time_spsa,
            "mem_spsa": mem_spsa,
            "time_hmc_true": time_hmc_true,
            "mem_hmc_true": mem_hmc_true,
            "time_hmc_estimated": time_hmc_estimated,
            "mem_hmc_estimated": mem_hmc_estimated,
            "time_hmc_spsa": time_hmc_spsa,
            "mem_hmc_spsa": mem_hmc_spsa,
            "all_steps_true": all_steps_true.tolist(),
            "all_steps_estimated": all_steps_estimated.tolist(),
            "all_steps_spsa": all_steps_spsa.tolist(),
            "all_steps_hmc_true": all_steps_hmc_true.tolist(),
            "all_steps_hmc_estimated": all_steps_hmc_estimated.tolist(),
            "all_steps_hmc_spsa": all_steps_hmc_spsa.tolist()
        })
        print(f"Completed: num_samples = {num_samples}, steps = {steps}")

# Save results to file
results_filename = os.path.join(results_dir, 'results_univariate.json')
with open(results_filename, 'w') as f:
    json.dump(results, f)
print(f"Results saved to {results_filename}")

colors = {
    'LD - True Gradients': 'blue',
    'LD - KW': 'green',
    'LD - SPSA': 'red',
    'HMC - True Gradients': 'purple',
    'HMC - KW': 'orange',
    'HMC - SPSA': 'brown'
}

x_vals = torch.linspace(-5, 5, 1000).to(device)
true_density = torch.exp(log_prob(x_vals)).detach().cpu().numpy()
# Plot convergence
def plot_convergence(results, num_samples, steps, filename):
    fig = go.Figure()
    
    for result in results:
        if result["num_samples"] == num_samples and result["steps"] == steps:
            samples_true = torch.tensor((result["all_steps_true"])).view(num_samples, steps)
            samples_estimated = torch.tensor((result["all_steps_estimated"])).view(num_samples, steps)
            samples_spsa = torch.tensor(result["all_steps_spsa"]).view(num_samples, steps)
            samples_hmc_true = torch.tensor(result["all_steps_hmc_true"]).view(num_samples, steps)
            samples_hmc_estimated = torch.tensor(result["all_steps_hmc_estimated"]).view(num_samples, steps)
            samples_hmc_spsa = torch.tensor(result["all_steps_hmc_spsa"]).view(num_samples, steps)
            
            for i in range(num_samples):
                fig.add_trace(go.Scatter(y=samples_true[i], mode='lines', name=f'LD - True Gradients', opacity=0.5, legendgroup='LD - True Gradients', line=dict(color=colors['LD - True Gradients']), showlegend=(i == 0)))
                fig.add_trace(go.Scatter(y=samples_estimated[i], mode='lines', name=f'LD - KW (num_sample={result["num_samples"]})', opacity=0.5, legendgroup='LD - KW', line=dict(color=colors['LD - KW']), showlegend=(i == 0)))
                fig.add_trace(go.Scatter(y=samples_spsa[i], mode='lines', name=f'LD - SPSA (num_sample={result["num_samples"]})', opacity=0.5, legendgroup='LD - SPSA', line=dict(color=colors['LD - SPSA']), showlegend=(i == 0)))
                fig.add_trace(go.Scatter(y=samples_hmc_true[i], mode='lines', name=f'HMC - True Gradients (num_sample={result["num_samples"]})', opacity=0.5, legendgroup='HMC - True Gradients', line=dict(color=colors['HMC - True Gradients']), showlegend=(i == 0)))
                fig.add_trace(go.Scatter(y=samples_hmc_estimated[i], mode='lines', name=f'HMC - KW (num_sample={result["num_samples"]})', opacity=0.5, legendgroup='HMC - KW', line=dict(color=colors['HMC - KW']), showlegend=(i == 0)))
                fig.add_trace(go.Scatter(y=samples_hmc_spsa[i], mode='lines', name=f'HMC - SPSA (num_sample={result["num_samples"]})', opacity=0.5, legendgroup='HMC - SPSA', line=dict(color=colors['HMC - SPSA']), showlegend=(i == 0)))
                
    fig.update_layout(title=f'Convergence Plot for steps={steps}',
                      xaxis_title='Steps',
                      yaxis_title='Sample Value',
                      legend_title_text='Methods')
    fig.write_html(filename)
    print(f"Plot saved to {filename}")

# Plot histogram comparison
def plot_histogram_comparison(results, num_samples, steps, filename):
    for result in results:
        if result["num_samples"] == num_samples and result["steps"] == steps:
            assert len(result["all_steps_true"]) == num_samples*steps
            samples_true = torch.tensor((result["all_steps_true"])).view(num_samples, steps)[:, -1]
            samples_estimated = torch.tensor((result["all_steps_estimated"])).view(num_samples, steps)[:, -1]
            samples_spsa = torch.tensor(result["all_steps_spsa"]).view(num_samples, steps)[:, -1]
            samples_hmc_true = torch.tensor(result["all_steps_hmc_true"]).view(num_samples, steps)[:, -1]
            samples_hmc_estimated = torch.tensor(result["all_steps_hmc_estimated"]).view(num_samples, steps)[:, -1]
            samples_hmc_spsa = torch.tensor(result["all_steps_hmc_spsa"]).view(num_samples, steps)[:, -1]

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=samples_true, nbinsx=100, histnorm='probability density', name='LD - True Gradients', opacity=0.5))
            fig.add_trace(go.Histogram(x=samples_estimated, nbinsx=100, histnorm='probability density', name='LD - KW', opacity=0.5))
            fig.add_trace(go.Histogram(x=samples_spsa, nbinsx=100, histnorm='probability density', name='LD - SPSA', opacity=0.5))
            fig.add_trace(go.Histogram(x=samples_hmc_true, nbinsx=100, histnorm='probability density', name='HMC - True Gradients', opacity=0.5))
            fig.add_trace(go.Histogram(x=samples_hmc_estimated, nbinsx=100, histnorm='probability density', name='HMC - KW', opacity=0.5))
            fig.add_trace(go.Histogram(x=samples_hmc_spsa, nbinsx=100, histnorm='probability density', name='HMC - SPSA', opacity=0.5))
            fig.add_trace(go.Scatter(x=x_vals.detach().cpu().numpy(), y=true_density, mode='lines', name='True Distribution', line=dict(color='black', width=2)))

            fig.update_layout(
                title=f'Histogram Comparison for num_samples={num_samples}, steps={steps}',
                xaxis_title='Sample Value',
                yaxis_title='Density',
                barmode='overlay'
            )

            fig.write_html(filename)
            print(f"Plot saved to {filename}")

num_samples = 10000
steps = 1000
# Example plot for specific num_samples and steps
plot_histogram_comparison(results, num_samples, steps, os.path.join(results_dir, f'histogram_comparison_{num_samples}_{steps}.html'))


num_samples = 1
steps = 1000
# Example plot for specific num_samples and steps
plot_convergence(results, num_samples, steps, os.path.join(results_dir, f'convergence_plot_{num_samples}_{steps}.html'))