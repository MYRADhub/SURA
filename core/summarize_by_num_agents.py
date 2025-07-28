import os
import argparse
import pandas as pd
import yaml
from collections import defaultdict

def get_num_agents(configs_dir, case_name):
    path = os.path.join(configs_dir, f"{case_name}.yaml")
    with open(path, 'r') as f:
        yml = yaml.safe_load(f)
    return len(yml.get("agents", []))

def aggregate_trials(df, key_col, value_col):
    """Average over multiple trials for each case, if needed."""
    if 'Trial' in df.columns:
        avg = df.groupby(key_col)[value_col].mean().reset_index()
        return dict(zip(avg[key_col], avg[value_col]))
    else:
        # No trial, just use as is
        return dict(zip(df[key_col], df[value_col]))

def main(result_csv, configs_dir, value_field=None):
    df = pd.read_csv(result_csv)
    # Detect field to use
    possible_fields = ["Steps", "Cost"]
    if value_field is None:
        for pf in possible_fields:
            if pf in df.columns:
                value_field = pf
                break
    if not value_field:
        raise ValueError(f"Could not detect field to summarize (fields: {df.columns})")

    # Aggregate if multi-trial
    val_map = aggregate_trials(df, "Case", value_field)

    # Map case -> num agents
    case2num = {}
    for case in val_map:
        try:
            n = get_num_agents(configs_dir, case)
            case2num[case] = n
        except Exception as e:
            print(f"⚠️ Could not load agent count for {case}: {e}")

    # Group by agent count
    by_agents = defaultdict(list)
    for case, val in val_map.items():
        n = case2num.get(case)
        if n is not None:
            by_agents[n].append(val)

    print(f"\nSummary for {os.path.basename(result_csv)}\nGrouped by number of agents:\n")
    print("{:<12} {:<10} {:<10} {:<10}".format("Agents", "Mean", "Std", "N"))
    summary_rows = []
    for n in sorted(by_agents):
        arr = by_agents[n]
        mean = sum(arr) / len(arr)
        std = (sum((x - mean) ** 2 for x in arr) / len(arr)) ** 0.5 if len(arr) > 1 else 0
        summary_rows.append((n, mean, std, len(arr)))
        print("{:<12} {:<10.2f} {:<10.2f} {:<10}".format(n, mean, std, len(arr)))

    summary_csv = os.path.splitext(result_csv)[0] + "_by_agents.csv"
    pd.DataFrame(summary_rows, columns=["Agents", "Mean", "Std", "N"]).to_csv(summary_csv, index=False)
    print(f"\nSaved summary to {summary_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize results by number of agents.")
    parser.add_argument("--csv", required=True, help="Input results CSV")
    parser.add_argument("--configs-dir", required=True, help="Directory containing YAML config files")
    parser.add_argument("--field", default=None, help="Value field to use (Steps, Cost, etc.)")
    args = parser.parse_args()
    main(args.csv, args.configs_dir, value_field=args.field)
