import csv
import os
import argparse
import importlib

MAX_STEPS = 50
IMAGE_PATH_TEMPLATE = "data/grid.png"

# Format: (task_key, module.path)
TASKS = [
    ("agent1", "agents.agent1"),
    ("agent1_obs", "agents.agent1_obs"),
    ("agent1_yesno", "agents.agent1_yesno"),
    ("agent1_uct", "agents.agent1_uct")
]

DIFFICULT_CASES = [
    ((4, 3), (0, 4)),
    ((3, 2), (2, 3)),
    ((2, 3), (3, 2)),
    ((1, 3), (4, 2)),
    ((2, 1), (4, 3)),
    ((2, 1), (4, 4)),
    ((2, 3), (5, 5)),
    ((0, 4), (3, 2)),
    ((0, 4), (4, 2)),
    ((0, 5), (4, 3)),
    ((0, 0), (5, 5)),
    ((5, 5), (0, 0)),
    ((0, 5), (5, 0)),
    ((5, 0), (0, 5)),
    ((3, 2), (0, 5)),
    ((3, 2), (5, 0)),
    ((0, 5), (3, 2)),
    ((5, 0), (3, 2)),
    ((1, 5), (0, 3)),
    ((0, 3), (1, 5)),
    ((0, 3), (5, 3)),
    ((5, 3), (0, 3)),
    ((5, 2), (1, 5)),
    ((1, 5), (5, 2)),
    ((4, 4), (0, 5)),
    ((5, 4), (0, 5)),
    ((0, 5), (4, 4)),
    ((0, 5), (5, 4)),
    ((2, 0), (5, 3)),
    ((5, 3), (2, 0)),
]

def evaluate_difficult(task_name, run_fn):
    print(f"\n=== Evaluating Difficult Cases: {task_name} ===")

    log_file = f"results/{task_name}_difficult_results.csv"
    summary_file = f"results/{task_name}_difficult_summary.txt"
    os.makedirs("results", exist_ok=True)
    write_header = not os.path.exists(log_file)

    total_steps = total_opt = fails = 0

    with open(log_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["Case", "Steps", "Optimal", "Failed"])

        for i, (start, goal) in enumerate(DIFFICULT_CASES):
            print(f"Case {i+1}: Agent {start} → Goal {goal}")
            steps, optimal, failed = run_fn(
                agent_start=start,
                goal_pos=goal,
                obstacles={(2, 2), (3, 3), (1, 4)},
                image_path=IMAGE_PATH_TEMPLATE.format(idx=i),
                max_steps=MAX_STEPS
            )
            writer.writerow([i + 1, steps, optimal, int(failed)])
            total_steps += steps
            total_opt += optimal
            fails += int(failed)

    avg_steps = total_steps / len(DIFFICULT_CASES)
    avg_opt = total_opt / len(DIFFICULT_CASES)
    diff = avg_steps - avg_opt
    percent = (diff / avg_opt) * 100 if avg_opt else 0

    summary_lines = [
        f"Evaluated on {len(DIFFICULT_CASES)} fixed difficult cases.",
        f"Average optimal path: {avg_opt:.2f}",
        f"Average steps taken: {avg_steps:.2f}",
        f"Difference: {diff:.2f} ({percent:.2f}%)",
        f"Failures (max steps): {fails}/{len(DIFFICULT_CASES)}",
        f"Results saved to {log_file}"
    ]

    print("\n".join(summary_lines))
    with open(summary_file, "w") as f:
        f.writelines(line + "\n" for line in summary_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agents on difficult GridWorld cases.")
    parser.add_argument(
        "--agents",
        type=str,
        choices=[task[0] for task in TASKS],
        nargs="+",
        default=[task[0] for task in TASKS],
        help="Specify which agents to evaluate (e.g. --agents agent1 agent1_obs)"
    )
    args = parser.parse_args()

    # Dynamically import run functions for selected agents
    selected_tasks = {key: mod for (key, mod) in TASKS if key in args.agents}
    for key, module_path in selected_tasks.items():
        module = importlib.import_module(module_path)
        run_fn = module.run
        evaluate_difficult(key, run_fn)
