import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_results(csv_path):
    """Load a CSV file with columns: Agents, Mean, Std, N"""
    df = pd.read_csv(csv_path)
    # Ensure sorting by agent count for clean x-axis
    df = df.sort_values(by='Agents')
    return df

def main(args):
    assert len(args.csvs) == len(args.labels), "Must provide one label per CSV."

    # Load all dataframes and collect unique agent counts
    all_dfs = []
    all_agent_counts = set()
    for csv in args.csvs:
        df = load_results(csv)
        all_dfs.append(df)
        all_agent_counts.update(df['Agents'])

    all_agent_counts = sorted(list(all_agent_counts))
    x = np.arange(len(all_agent_counts))  # x locations for the groups

    # Bar width and figure setup
    total_width = 0.8
    n_bars = len(args.csvs)
    bar_width = total_width / n_bars

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each CSV as a group of bars
    for i, (df, label) in enumerate(zip(all_dfs, args.labels)):
        # Align each method's means and stds with the full list of agent counts
        means = []
        stds = []
        for a in all_agent_counts:
            row = df[df['Agents'] == a]
            means.append(row['Mean'].values[0] if not row.empty else np.nan)
            stds.append(row['Std'].values[0] if not row.empty else 0)
        # Bar positions, shifted for grouping
        positions = x - total_width/2 + i*bar_width + bar_width/2
        ax.bar(positions, means, width=bar_width, yerr=stds, capsize=6, label=label)

    # Labeling and ticks
    ax.set_xlabel('Number of agents')
    ax.set_ylabel('Mean performance')
    ax.set_xticks(x)
    ax.set_xticklabels(all_agent_counts)
    ax.set_title('Performance by Number of Agents')
    ax.legend()
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out)
        print(f"Plot saved to {args.out}")
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare baselines with grouped bar plot.')
    parser.add_argument('--csvs', nargs='+', required=True, help='Paths to CSV files (one per method)')
    parser.add_argument('--labels', nargs='+', required=True, help='Custom labels (one per CSV)')
    parser.add_argument('--out', type=str, default=None, help='Output file for figure (e.g., plot.pdf)')
    args = parser.parse_args()
    main(args)
