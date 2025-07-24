import argparse
import csv
import os
import random
from core.environment import GridWorld
from core.utils import shortest_path_length

def compute_random_assignment_cost(config_path):
    env = GridWorld(config_path)

    agents = [a for a in env.agents if a is not None]
    goals = [g for g in env.goals if g is not None]

    if len(agents) != len(goals):
        raise ValueError("Number of agents and goals must be equal.")

    goal_indices = list(range(len(goals)))
    random.shuffle(goal_indices)

    assigned_costs = []
    for i, agent_pos in enumerate(agents):
        goal_pos = goals[goal_indices[i]]
        dist = shortest_path_length(agent_pos, goal_pos, env)
        assigned_costs.append(dist)

    team_cost = max(assigned_costs)
    return team_cost

if __name__ == "__main__":
    random.seed(42)

    parser = argparse.ArgumentParser(description="Compute team cost from random agent-goal assignment.")
    parser.add_argument("--config", type=str, help="Path to a single YAML config file")
    parser.add_argument("--configs-dir", type=str, help="Path to a directory of YAML config files")
    parser.add_argument("--output", type=str, default="random_baseline.csv", help="CSV file to store results")

    args = parser.parse_args()

    if bool(args.config) == bool(args.configs_dir):  # both or neither
        raise ValueError("You must provide exactly one of --config or --configs-dir.")

    rows = []

    if args.config:
        cost = compute_random_assignment_cost(args.config)
        case_name = os.path.splitext(os.path.basename(args.config))[0]
        rows.append([case_name, cost])
        print(f"✅ {case_name}: Random assignment cost = {cost}")

    else:
        for filename in sorted(os.listdir(args.configs_dir)):
            if filename.endswith(".yaml"):
                config_path = os.path.join(args.configs_dir, filename)
                case_name = os.path.splitext(filename)[0]
                try:
                    cost = compute_random_assignment_cost(config_path)
                    rows.append([case_name, cost])
                    print(f"✅ {case_name}: Random assignment cost = {cost}")
                except Exception as e:
                    print(f"❌ Error in {case_name}: {e}")

    if rows:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Case", "RandomCost"])
            writer.writerows(rows)

        print(f"\n📄 Random assignment results saved to: {args.output}")
