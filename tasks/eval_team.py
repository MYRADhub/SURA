import csv
import os
import argparse
import importlib
from core.environment import GridWorld
from core.plot import plot_grid_unassigned

# Constants
IMAGE_PATH = "data/grid.png"
OUTPUT_DIR = "results_team"
VISUALIZE = False  # set to True if you want to save visuals
GRID_SIZE = 8
NUM_AGENTS = 3
TRIALS_PER_CASE = 2

# Multi-agent systems to evaluate
TASKS = [
    ("agent_yesno_unassigned", "agents.agent_yesno_unassigned"),
    ("agent_com_unassigned", "agents.agent_com_unassigned"),
    ("agent_com_unstruc", "agents.agent_com_unstruc"),
    ("agent_collab", "agents.agent_collab"),
]

# Shared obstacle layout (no overlaps)
obstacles = {(3, 3), (4, 4), (2, 5), (5, 2), (6, 6)}

# Hand-crafted coordination scenarios
cases = {
    "case_1": "configs/case_1_local_greed.yaml",
    "case_2": "configs/case_2_decoy_efficiency.yaml",
    "case_3": "configs/case_3_one_stays_back.yaml",
    "case_4": "configs/case_4_tight_corridor.yaml",
    "case_5": "configs/case_5_decoy_ignored.yaml",
    "case_6": "configs/case_6_lane_assignment.yaml",
    "case_7": "configs/case_7_vertical_choke.yaml",
    "case_8": "configs/case_8_goal_stealing.yaml",
    "case_9": "configs/case_9_2_greedy_agents.yaml",
    "case_10": "configs/case_10_insane.yaml"
}

def evaluate_team(task_key, run_fn):
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
                    plot_grid_unassigned(env, image_path=f"data/{task_key}_{case_name}_t{trial+1}.png")

                case_log_path = os.path.join(OUTPUT_DIR, f"{task_key}_{case_name}_trial{trial+1}_log.csv")

                steps, optimal, failed, collisions = run_fn(
                    config_path=case_config_path,
                    log_path=case_log_path,
                    image_path=IMAGE_PATH,
                    max_steps=30
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
    parser.add_argument("--visualize", action="store_true", help="Save visualizations for each scenario")

    args = parser.parse_args()
    VISUALIZE = args.visualize

    selected_agents = {key: mod for (key, mod) in TASKS if key in args.agents}
    for key, module_path in selected_agents.items():
        module = importlib.import_module(module_path)
        run_fn = module.run
        evaluate_team(key, run_fn)
