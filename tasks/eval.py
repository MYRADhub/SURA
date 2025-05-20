import csv
import os
import argparse
from tasks.agent1 import run as run_1_agent
from tasks.agent1_obs import run as run_1_agent_obs
from tasks.agents2 import run as run_2_agents
from tasks.agents2_obs import run as run_2_agents_obs

GRID_SIZE = 6
NUM_RUNS = 20
MAX_STEPS = 30
IMAGE_PATH = "data/grid.png"

def extract_direction(response):
    for d in ['up', 'down', 'left', 'right']:
        if d in response.lower():
            return d
    return None

def evaluate(task_name, runner, is_two_agents=False):
    print(f"\n=== Evaluating {task_name} ===")
    total_steps = total_opt = fails = total_collisions = 0
    log_file = f"results/{task_name}_results.csv"
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
            result = runner()
            if is_two_agents:
                steps, optimal, failed, collisions = result
                total_collisions += collisions
                writer.writerow([i+1, steps, optimal, int(failed), collisions])
            else:
                steps, optimal, failed = result
                writer.writerow([i+1, steps, optimal, int(failed)])

            total_steps += steps
            total_opt += optimal
            if failed:
                fails += 1

    avg_steps = total_steps / NUM_RUNS
    avg_opt = total_opt / NUM_RUNS
    diff = avg_steps - avg_opt
    percent = (diff / avg_opt) * 100 if avg_opt else 0

    print(f"\n== {task_name} Summary ==")
    print(f"Average optimal path: {avg_opt:.2f}")
    print(f"Average steps taken : {avg_steps:.2f}")
    print(f"Difference          : {diff:.2f} ({percent:.2f}%)")
    print(f"Failures (max steps): {fails}/{NUM_RUNS}")
    if is_two_agents:
        print(f"Collisions          : {total_collisions} total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GridWorld agents.")
    parser.add_argument(
        "--agents",
        type=str,
        choices=["1_agent", "1_agent_obs", "2_agents", "2_agents_obs"],
        nargs="+",
        default=["1_agent", "1_agent_obs", "2_agents", "2_agents_obs"],
        help="Specify which agents to evaluate (space separated, e.g. --agents 1_agent 2_agents)."
    )
    args = parser.parse_args()

    agent_configs = {
        "1_agent": (run_1_agent, False),
        "1_agent_obs": (run_1_agent_obs, False),
        "2_agents": (run_2_agents, True),
        "2_agents_obs": (run_2_agents_obs, True),
    }

    for agent in args.agents:
        runner, is_two_agents = agent_configs[agent]
        evaluate(agent, runner, is_two_agents)
