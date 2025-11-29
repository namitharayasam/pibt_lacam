import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import os

def plot_success_rate(csv_path: str, output_plot: str = None):
    # Read data
    data = pd.read_csv(csv_path)
    
    # Calculate success rate for each algorithm and agent count
    success_rates = data.groupby(['algorithm', 'num_agents'])['success'].mean().reset_index()
    
    # Create plot
    FONT_SIZE = 10
    plt.rcParams.update({'font.size': FONT_SIZE})
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot each algorithm
    for algo in success_rates['algorithm'].unique():
        algo_data = success_rates[success_rates['algorithm'] == algo]
        ax.plot(algo_data['num_agents'], algo_data['success'], 
               marker='o', markersize=8, label=algo, linewidth=2)
    
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Number of Agents', fontsize=FONT_SIZE + 2)
    ax.set_ylabel('Success Rate', fontsize=FONT_SIZE + 2)
    ax.tick_params(axis='both', labelsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE, frameon=True)
    ax.grid(True, alpha=0.3)
    
    # Save plot
    if output_plot is None:
        output_plot = csv_path.replace('.csv', '_success_rate.png')
    
    plt.savefig(output_plot, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Plot saved to: {output_plot}")

def plot_computation_time(csv_path: str, output_plot: str = None):
    data = pd.read_csv(csv_path)
    
    # Only consider successful runs
    successful_data = data[data['success'] == True]
    
    if len(successful_data) == 0:
        print("No successful runs to plot computation time")
        return
    
    # Calculate average time
    avg_times = successful_data.groupby(['algorithm', 'num_agents'])['time'].mean().reset_index()
    
    FONT_SIZE = 10
    plt.rcParams.update({'font.size': FONT_SIZE})
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for algo in avg_times['algorithm'].unique():
        algo_data = avg_times[avg_times['algorithm'] == algo]
        ax.plot(algo_data['num_agents'], algo_data['time'], 
               marker='s', markersize=8, label=algo, linewidth=2)
    
    ax.set_xlabel('Number of Agents', fontsize=FONT_SIZE + 2)
    ax.set_ylabel('Average Computation Time (s)', fontsize=FONT_SIZE + 2)
    ax.tick_params(axis='both', labelsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE, frameon=True)
    ax.grid(True, alpha=0.3)
    
    if output_plot is None:
        output_plot = csv_path.replace('.csv', '_time.png')
    
    plt.savefig(output_plot, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Plot saved to: {output_plot}")

def plot_solution_quality(csv_path: str, output_plot: str = None):
    data = pd.read_csv(csv_path)
    
    # Only consider successful runs
    successful_data = data[data['success'] == True]
    
    if len(successful_data) == 0:
        print("No successful runs to plot solution quality")
        return
    
    # Calculate average SOC
    avg_soc = successful_data.groupby(['algorithm', 'num_agents'])['soc'].mean().reset_index()
    
    FONT_SIZE = 10
    plt.rcParams.update({'font.size': FONT_SIZE})
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for algo in avg_soc['algorithm'].unique():
        algo_data = avg_soc[avg_soc['algorithm'] == algo]
        ax.plot(algo_data['num_agents'], algo_data['soc'], 
               marker='^', markersize=8, label=algo, linewidth=2)
    
    ax.set_xlabel('Number of Agents', fontsize=FONT_SIZE + 2)
    ax.set_ylabel('Average Sum-of-Costs', fontsize=FONT_SIZE + 2)
    ax.tick_params(axis='both', labelsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE, frameon=True)
    ax.grid(True, alpha=0.3)
    
    if output_plot is None:
        output_plot = csv_path.replace('.csv', '_soc.png')
    
    plt.savefig(output_plot, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Plot saved to: {output_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot comparison results')
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV file with results')
    parser.add_argument('--metrics', type=str, nargs='+',
                       default=['success', 'time', 'soc'],
                       choices=['success', 'time', 'soc'],
                       help='Metrics to plot')
    
    args = parser.parse_args()
    
    print(f"\nGenerating plots from: {args.csv}")
    
    if 'success' in args.metrics:
        plot_success_rate(args.csv)
    
    if 'time' in args.metrics:
        plot_computation_time(args.csv)
    
    if 'soc' in args.metrics:
        plot_solution_quality(args.csv)
    
    print("\nAll plots generated!")