import csv
import os
import argparse
import importlib
from core.environment import GridWorld
from core.plot import plot_grid_unassigned_labeled

# Constants
IMAGE_PATH = "data/grid.png"
OUTPUT_DIR = "results_team"
VISUALIZE = False  # set to True if you want to save visuals
TRIALS_PER_CASE = 2

# Multi-agent systems to evaluate
TASKS = [
    ("agent_yesno_unassigned", "agents.agent_yesno_unassigned"),
    ("agent_com_unassigned", "agents.agent_com_unassigned"),
    ("agent_com_unstruc", "agents.agent_com_unstruc"),
    ("agent_collab", "agents.agent_collab"),
    ("agent_rank", "agents.agent_rank"),
    ("agent_rank_top2", "agents.agent_rank_top2"),
    ("agent_rank_priority_bfs", "agents.agent_rank_priority_bfs"),
    ("agent_rank_priority_llmdir", "agents.agent_rank_priority_llmdir"),
    ("agent_rank_once_bfs", "agents.agent_rank_once_bfs"),
    ("agent_rank_once_bfs_o3", "agents.agent_rank_once_bfs_o3"),
    ("agent_final_no_distances", "agents.agent_final_no_distances")
]

def evaluate_team(task_key, run_fn, cases):
    print(f"\n=== Evaluating Structured Cases: {task_key} ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, f"{task_key}_team_results.csv")
    summary_path = os.path.join(OUTPUT_DIR, f"{task_key}_team_summary.txt")

    write_header = not os.path.exists(log_path)
    total_steps = total_opt = fails = total_collisions = 0
    total_trials = 0

    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Case", "Trial", "Steps", "Optimal", "Failed", "Collisions"])

        for case_name, cfg in cases.items():
            for trial in range(TRIALS_PER_CASE):
                print(f"\n--- {case_name}, Trial {trial+1} ---")
                case_config_path = cfg  # it's already a path string to YAML

                if VISUALIZE:
                    env = GridWorld(case_config_path)
                    plot_grid_unassigned_labeled(env, image_path=f"data/{task_key}_{case_name}_t{trial+1}.png")

                case_log_path = os.path.join(OUTPUT_DIR, f"{task_key}_{case_name}_trial{trial+1}_log.csv")

                steps, optimal, failed, collisions = run_fn(
                    config_path=case_config_path,
                    log_path=case_log_path,
                    image_path=IMAGE_PATH,
                    max_steps=100
                )

                writer.writerow([case_name, trial+1, steps, optimal, int(failed), collisions])
                total_steps += steps
                total_opt += optimal
                fails += int(failed)
                total_collisions += collisions
                total_trials += 1

    avg_steps = total_steps / total_trials
    avg_opt = total_opt / total_trials
    diff = avg_steps - avg_opt
    percent = (diff / avg_opt) * 100 if avg_opt else 0

    summary_lines = [
        f"Evaluated {total_trials} trials across {len(cases)} scenarios",
        f"Average optimal path: {avg_opt:.2f}",
        f"Average steps taken: {avg_steps:.2f}",
        f"Difference: {diff:.2f} ({percent:.2f}%)",
        f"Failures (max steps): {fails}/{total_trials}",
        f"Total collisions: {total_collisions}",
    ]

    print("\n".join(summary_lines))
    with open(summary_path, "w") as f:
        f.writelines(line + "\n" for line in summary_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multi-agent systems on coordination scenarios.")
    parser.add_argument("--agents", type=str, choices=[t[0] for t in TASKS], nargs="+", default=[t[0] for t in TASKS])
    parser.add_argument("--config-dir", type=str, default="configs/difficult", help="Directory containing configuration YAML files")
    parser.add_argument("--visualize", action="store_true", help="Save visualizations for each scenario")
    parser.add_argument("--trials", type=int, default=TRIALS_PER_CASE, help="Number of trials per case")
    args = parser.parse_args()

    VISUALIZE = args.visualize
    TRIALS_PER_CASE = args.trials

    # Load all YAML cases
    cases = {
        os.path.splitext(f)[0]: os.path.join(args.config_dir, f)
        for f in os.listdir(args.config_dir)
        if f.endswith(".yaml")
    }
    if not cases:
        raise ValueError(f"No YAML files found in directory: {args.config_dir}")

    print(f"Found {len(cases)} cases in {args.config_dir}")
    print("Cases:", list(cases.keys())[:10])

    # Prepare mapping of selected agents
    selected_agents = {key: mod for (key, mod) in TASKS if key in args.agents}

    # Load completed trials for each agent
    completed = {key: set() for key in selected_agents}
    for key in selected_agents:
        log_path = os.path.join(OUTPUT_DIR, f"{key}_team_results.csv")
        if os.path.exists(log_path):
            with open(log_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    completed[key].add((row["Case"], int(row["Trial"])))

    print("Completed trials for agents:")
    for key, trials in completed.items():
        print(f"{key}: {len(trials)} trials")

    input("Found completed trials for agents. Press Enter to continue...")

    # Run evaluation for each agent
    for key, module_path in selected_agents.items():
        # Filter out cases where all trials are done (trial 1 and 2)
        agent_cases = {
            case: path
            for case, path in cases.items()
            if (case, 1) not in completed[key] or (case, 2) not in completed[key]
        }

        print(f"\n{key} has {len(agent_cases)} cases left to run.")
        if not agent_cases:
            print(f"✅ Skipping {key} — all trials completed.")
            continue

        module = importlib.import_module(module_path)
        run_fn = module.run
        evaluate_team(key, run_fn, cases=agent_cases)

