import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

def load_results(file_path):
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results

def plot_results(results, output_dir):
    plot_configs = [
        ('train_losses', 'Train Loss', 'Train Loss vs Epoch for Different Optimizers'),
        ('test_losses', 'Test Loss', 'Test Loss vs Epoch for Different Optimizers'),
        # ('train_accuracies', 'Train Accuracy', 'Train Accuracy vs Epoch for Different Optimizers'),
        # ('test_accuracies', 'Test Accuracy', 'Test Accuracy vs Epoch for Different Optimizers'),
        ('compute_costs', 'Compute Cost (MB)', 'Compute Cost vs Epoch for Different Optimizers'),
        # ('complexities', 'Complexity', 'Complexity vs Epoch for Different Optimizers'),
        ('times', 'Time (seconds)', 'Time vs Epoch for Different Optimizers'),
    ]

    fig = go.Figure()
    buttons = []
    traces = {key: [] for key, _, _ in plot_configs}

    for result_type, metric_label, title in plot_configs:
        for optimizer, metrics in results.items():
            if optimizer == 'true_params':
                continue
            epochs = range(1, len(metrics[result_type]) + 1)
            trace = go.Scatter(
                x=list(epochs), y=metrics[result_type],
                mode='lines', name=f"{metric_label} - {optimizer}",
                visible=False
            )
            traces[result_type].append(trace)
            fig.add_trace(trace)

    for result_type, metric_label, title in plot_configs:
        button = dict(
            label=metric_label,
            method='update',
            args=[{
                'visible': [trace in traces[result_type] for trace in fig.data]
            }, {
                'title': title
            }]
        )
        buttons.append(button)

    if buttons:
        first_metric = plot_configs[0][0]
        buttons[0]['args'][0]['visible'] = [trace in traces[first_metric] for trace in fig.data]

    fig.update_layout(
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
        }],
        title='Metrics for Different Optimizers',
        xaxis_title='Epoch',
        yaxis_title='Value'
    )

    fig.write_html(output_dir / 'metrics_evolution.html')

def plot_parameter_evolution(results, output_dir):
    param_hist_all = {}
    for optimizer, metrics in results.items():
        if optimizer == 'true_params':
            continue
        if 'param_hist' in metrics:
            param_hist_all[optimizer] = metrics['param_hist']
        else:
            print(f"Warning: 'param_hist' key is missing for optimizer {optimizer}")

    true_params = results["true_params"]
    
    fig = go.Figure()
    buttons = []
    param_traces = []
    for param_name, true_value_dict in true_params.items():
        for param_key, true_value in true_value_dict.items():
            df = pd.DataFrame()
            for optimizer, param_hist in param_hist_all.items():
                if param_key in param_hist:
                    param_values = param_hist[param_key]
                    epochs = range(1, len(param_values) + 1)
                    df_temp = pd.DataFrame({'Epoch': epochs, 'Value': param_values, 'Optimizer': optimizer})
                    df = pd.concat([df, df_temp], ignore_index=True)
                else:
                    print(f"Error: param_hist_key {param_key} not found for optimizer {optimizer}")

            if not df.empty:
                for optimizer in df['Optimizer'].unique():
                    df_optimizer = df[df['Optimizer'] == optimizer]
                    trace = go.Scatter(
                        x=df_optimizer['Epoch'], y=df_optimizer['Value'],
                        mode='lines', name=f"{param_key} - {optimizer}",
                        visible=False
                    )
                    fig.add_trace(trace)
                    param_traces.append(trace)
                true_trace = go.Scatter(
                    x=[1, len(epochs)], y=[true_value]*2,
                    mode='lines', name=f"{param_key} - True Value",
                    line=dict(dash='dash'), visible=False
                )
                fig.add_trace(true_trace)
                param_traces.append(true_trace)

    for param_key in true_params[list(true_params.keys())[0]].keys():
        button = dict(
            label=param_key,
            method='update',
            args=[{
                'visible': [trace.name.startswith(f"{param_key} -") for trace in fig.data]
            }, {
                'title': f'Evolution of {param_key} for Different Optimizers'
            }]
        )
        buttons.append(button)
    
    # Set the initial visibility for the first parameter
    if buttons:
        buttons[0]['args'][0]['visible'] = [trace.name.startswith(f"{buttons[0]['label']} -") for trace in fig.data]

    fig.update_layout(
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
        }],
        title='Evolution of Parameters for Different Optimizers',
        xaxis_title='Epoch',
        yaxis_title='Parameter Value'
    )

    fig.write_html(output_dir / 'parameters_evolution.html')

def main():
    results_file = 'optimizer_results.json'
    results = load_results(results_file)

    output_dir = Path('optimizer_plots/linear_regression/aggregated_plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_parameter_evolution(results, output_dir)
    plot_results(results, output_dir)

if __name__ == '__main__':
    main()