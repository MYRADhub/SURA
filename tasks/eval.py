import csv
import os
import argparse
import importlib

GRID_SIZE = 6
NUM_RUNS = 50
MAX_STEPS = 30
IMAGE_PATH = "data/grid.png"

# Add new tasks here: ("task_key", "module.path", is_two_agents)
TASKS = [
    ("agent1", "tasks.agent1", False),
    ("agent1_obs", "tasks.agent1_obs", False),
    ("agents2", "tasks.agents2", True),
    ("agents2_obs", "tasks.agents2_obs", True),
    ("agent1_uct", "tasks.agent1_uct", False)
]


def extract_direction(response):
    for d in ['up', 'down', 'left', 'right']:
        if d in response.lower():
            return d
    return None

def evaluate(task_name, runner, is_two_agents=False):
    print(f"\n=== Evaluating {task_name} ===")
    total_steps = total_opt = fails = total_collisions = 0
    llm_disagree_total = 0
    log_file = f"results/{task_name}_results.csv"
    summary_file = f"results/{task_name}_summary.txt"
    if not os.path.exists("results"):
        os.makedirs("results")
    write_header = not os.path.exists(log_file)

    with open(log_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            if is_two_agents:
                writer.writerow(["Episode", "Steps", "Optimal", "Failed", "Collisions"])
            else:
                writer.writerow(["Episode", "Steps", "Optimal", "Failed"])
        for i in range(NUM_RUNS):
            print(f"Episode {i+1}/{NUM_RUNS}...")
            # Handle UCT tasks with llm_disagree_count
            if "uct" in task_name:
                if is_two_agents:
                    steps, optimal, failed, collisions, llm_disagree_count = runner()
                    total_collisions += collisions
                else:
                    steps, optimal, failed, llm_disagree_count = runner()
                llm_disagree_total += llm_disagree_count
            else:
                if is_two_agents:
                    steps, optimal, failed, collisions = result = runner()
                    total_collisions += collisions
                else:
                    steps, optimal, failed = result = runner()
            if is_two_agents:
                writer.writerow([i+1, steps, optimal, int(failed), total_collisions])
            else:
                writer.writerow([i+1, steps, optimal, int(failed)])

            total_steps += steps
            total_opt += optimal
            if failed:
                fails += 1

    avg_steps = total_steps / NUM_RUNS
    avg_opt = total_opt / NUM_RUNS
    diff = avg_steps - avg_opt
    percent = (diff / avg_opt) * 100 if avg_opt else 0

    summary_lines = [
        f"Average optimal path: {avg_opt:.2f}",
        f"Average steps taken: {avg_steps:.2f}",
        f"Difference: {diff:.2f} ({percent:.2f}%)",
        f"Failures (max steps): {fails}/{NUM_RUNS}"
    ]
    if is_two_agents:
        summary_lines.append(f"Collisions: {total_collisions} total")
    if "uct" in task_name:
        summary_lines.append(f"LLM/UCT disagreement count: {llm_disagree_total}")
    summary_lines.append(f"Results saved to {log_file}")

    print(f"\n{task_name} Summary")
    for line in summary_lines:
        print(f"{line}")

    with open(summary_file, 'w') as f:
        for line in summary_lines:
            f.write(f"{line}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GridWorld agents.")
    parser.add_argument(
        "--agents",
        type=str,
        choices=[task[0] for task in TASKS],
        nargs="+",
        default=[task[0] for task in TASKS],
        help="Specify which agents to evaluate (space separated, e.g. --agents agent1 agents2)."
    )
    args = parser.parse_args()

    # Dynamically import run functions
    agent_configs = {}
    for key, module_path, is_two_agents in TASKS:
        module = importlib.import_module(module_path)
        agent_configs[key] = (module.run, is_two_agents)

    for agent in args.agents:
        runner, is_two_agents = agent_configs[agent]
        evaluate(agent, runner, is_two_agents)
