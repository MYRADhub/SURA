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
]

# Shared obstacle layout (no overlaps)
obstacles = {(3, 3), (4, 4), (2, 5), (5, 2), (6, 6)}

# Hand-crafted coordination scenarios
cases = {
    "case_1_local_greed": {
        "agents": [(3, 1), (3, 5)],
        "goals":  [(1, 4), (0, 6)],
    },
    "case_2_decoy_efficiency": {
        "agents": [(0, 0), (7, 0)],
        "goals":  [(7, 7), (3, 4)],
    },
    "case_3_one_stays_back": {
        "agents": [(1, 1), (2, 1), (7, 6)],
        "goals":  [(6, 7), (1, 7), (3, 6)],
    },
    "case_4_tight_corridor": {
        "agents": [(0, 1), (1, 0), (0, 0)],
        "goals":  [(6, 1), (7, 2), (7, 1)],
    },
    "case_5_decoy_ignored": {
        "agents": [(7, 0), (6, 0), (0, 7)],
        "goals":  [(0, 0), (1, 1), (2, 2)],
    },
    "case_6_lane_assignment": {
        "agents": [(2, 2), (2, 3), (2, 4)],
        "goals":  [(6, 2), (6, 3), (6, 4)],
    },
    "case_7_vertical_choke": {
        "agents": [(0, 3), (0, 5), (0, 7)],
        "goals":  [(7, 3), (7, 5), (7, 7)],
    },
    "case_8_goal_stealing": {
        "agents": [(6, 0), (6, 1), (6, 2)],
        "goals":  [(0, 0), (0, 1), (0, 2)],
    }
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
                env = GridWorld(GRID_SIZE, obstacles=obstacles)
                env.initialize_agents_goals_custom(agents=cfg["agents"], goals=cfg["goals"])

                if VISUALIZE:
                    plot_grid_unassigned(env, image_path=f"data/{task_key}_{case_name}_t{trial+1}.png")

                steps, optimal, failed, collisions = run_fn(
                    grid_size=GRID_SIZE,
                    obstacles=obstacles,
                    agent_starts=cfg["agents"],
                    goal_positions=cfg["goals"],
                    image_path=IMAGE_PATH,
                    max_steps=30,
                    num_agents=len(cfg["agents"])
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
