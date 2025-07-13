import argparse
import csv
import os
from core.environment import GridWorld
from core.find_optim_sol import compute_distance_matrix, find_best_assignment

def compute_optimal_cost(config_path):
    env = GridWorld(config_path)
    distances = compute_distance_matrix(env)
    _, cost = find_best_assignment(distances)
    return cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute optimal goal assignment costs.")
    parser.add_argument("--config", type=str, help="Path to a single YAML config file")
    parser.add_argument("--configs-dir", type=str, help="Path to a directory of YAML config files")
    parser.add_argument("--output", type=str, default="optim_summary.csv", help="CSV file to store optimal results")

    args = parser.parse_args()

    if bool(args.config) == bool(args.configs_dir):  # both or neither set
        raise ValueError("You must provide exactly one of --config or --configs-dir.")

    rows = []

    if args.config:
        cost = compute_optimal_cost(args.config)
        case_name = os.path.splitext(os.path.basename(args.config))[0]
        rows.append([case_name, cost])
        print(f"✅ {case_name}: Optimal cost = {cost}")

    else:
        for filename in sorted(os.listdir(args.configs_dir)):
            if filename.endswith(".yaml"):
                config_path = os.path.join(args.configs_dir, filename)
                case_name = os.path.splitext(filename)[0]
                try:
                    cost = compute_optimal_cost(config_path)
                    rows.append([case_name, cost])
                    print(f"✅ {case_name}: Optimal cost = {cost}")
                except Exception as e:
                    print(f"❌ Error in {case_name}: {e}")

    if rows:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Case", "OptimalCost"])
            writer.writerows(rows)

        print(f"\n📄 Optimal costs saved to: {args.output}")
