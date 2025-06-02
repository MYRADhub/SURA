import csv
import os
import argparse
import importlib
import random
from core.utils import is_reachable

MAX_STEPS = 100
GRID_SIZE = 20
TRIALS_PER_CONFIG = 2
RANDOM_SEED = 42
IMAGE_PATH_TEMPLATE = "data/grid.png"

# (task_key, module_path)
TASKS = [
    ("agent1", "tasks.agent1"),
    ("agent1_obs", "tasks.agent1_obs"),
    ("agent1_yesno", "tasks.agent1_yesno"),
    ("agent1_uct", "tasks.agent1_uct"),
]

NUM_OBSTACLE_WORLDS = 5
AGENT_GOAL_PAIRS_PER_WORLD = 2

def generate_random_obstacles(grid_size, num_obstacles=20, avoid=set()):
    obstacles = set()
    while len(obstacles) < num_obstacles:
        r = random.randint(0, grid_size - 1)
        c = random.randint(0, grid_size - 1)
        cell = (r, c)
        if cell not in avoid:
            obstacles.add(cell)
    return obstacles

def generate_random_positions(grid_size, count, avoid=set()):
    positions = set()
    while len(positions) < count:
        pos = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        if pos not in avoid:
            positions.add(pos)
    return list(positions)

def evaluate_random(task_key, run_fn):
    print(f"\n=== Evaluating Random Worlds: {task_key} ===")
    os.makedirs("results", exist_ok=True)
    log_file = f"results/{task_key}_random_results.csv"
    summary_file = f"results/{task_key}_random_summary.txt"
    write_header = not os.path.exists(log_file)

    total_steps = total_opt = fails = 0
    total_cases = 0

    random.seed(RANDOM_SEED)
    with open(log_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["World", "Pair", "Trial", "Steps", "Optimal", "Failed"])

        for w in range(NUM_OBSTACLE_WORLDS):
            print(f"\n--- Obstacle World {w+1} ---")
            # Temporarily generate dummy positions to avoid placing obstacles there
            dummy_positions = {(0, 0), (GRID_SIZE - 1, GRID_SIZE - 1)}
            obstacles = generate_random_obstacles(GRID_SIZE, avoid=dummy_positions)

            # Generate valid agent-goal pairs that don’t overlap with obstacles and are reachable
            pairs = []
            attempts = 0
            while len(pairs) < AGENT_GOAL_PAIRS_PER_WORLD and attempts < 100:
                agent_pos, goal_pos = generate_random_positions(GRID_SIZE, 2, avoid=obstacles)
                if agent_pos != goal_pos and is_reachable(GRID_SIZE, agent_pos, goal_pos, obstacles):
                    pairs.append((agent_pos, goal_pos))
                attempts += 1

            for p, (agent_pos, goal_pos) in enumerate(pairs):

                for t in range(TRIALS_PER_CONFIG):
                    print(f"World {w+1}, Pair {p+1}, Trial {t+1} — Agent {agent_pos} → Goal {goal_pos}")
                    steps, optimal, failed = run_fn(
                        grid_size=GRID_SIZE,
                        agent_start=agent_pos,
                        goal_pos=goal_pos,
                        obstacles=obstacles,
                        image_path=IMAGE_PATH_TEMPLATE,
                        max_steps=MAX_STEPS
                    )
                    writer.writerow([w+1, p+1, t+1, steps, optimal, int(failed)])
                    total_steps += steps
                    total_opt += optimal
                    fails += int(failed)
                    total_cases += 1

    avg_steps = total_steps / total_cases
    avg_opt = total_opt / total_cases
    diff = avg_steps - avg_opt
    percent = (diff / avg_opt) * 100 if avg_opt else 0

    summary_lines = [
        f"Evaluated {total_cases} trials across 5 random obstacle worlds × 2 random agent-goal pairs × 2 trials",
        f"Average optimal path: {avg_opt:.2f}",
        f"Average steps taken: {avg_steps:.2f}",
        f"Difference: {diff:.2f} ({percent:.2f}%)",
        f"Failures (max steps): {fails}/{total_cases}",
        f"Results saved to {log_file}"
    ]

    print("\n".join(summary_lines))
    with open(summary_file, "w") as f:
        f.writelines(line + "\n" for line in summary_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agents on random GridWorlds.")
    parser.add_argument(
        "--agents",
        type=str,
        choices=[t[0] for t in TASKS],
        nargs="+",
        default=[t[0] for t in TASKS],
        help="Which agents to evaluate (e.g. --agents agent1 agent1_yesno)"
    )
    args = parser.parse_args()

    selected_tasks = {key: mod for (key, mod) in TASKS if key in args.agents}
    for key, module_path in selected_tasks.items():
        module = importlib.import_module(module_path)
        run_fn = module.run
        evaluate_random(key, run_fn)
