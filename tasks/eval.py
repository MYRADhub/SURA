import csv
import os
import argparse
import importlib
import random
from core.utils import is_reachable

MAX_STEPS = 30
GRID_SIZE = 6
TRIALS_PER_CONFIG = 2
RANDOM_SEED = 42
IMAGE_PATH = "data/grid.png"
NUM_OBSTACLE_WORLDS = 5
AGENT_GOAL_PAIRS_PER_WORLD = 2
NUM_OBSTACLES = 3
NUM_AGENTS = 3

# (task_key, module_path)
TASKS = [
    ("agent1", "agents.agent1"),
    ("agent1_obs", "agents.agent1_obs"),
    ("agent1_yesno", "agents.agent1_yesno"),
    ("agent1_uct", "agents.agent1_uct"),
    ("agent1_code", "agents.agent1_code"),
    ("agent_multi_yesno", "agents.agent_multi_yesno"),
    ("agent_yesno_unassigned", "agents.agent_yesno_unassigned"),
    ("agent_com_unassigned", "agents.agent_com_unassigned"),
]

def generate_random_obstacles(grid_size, num_obstacles=NUM_OBSTACLES, avoid=set()):
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

    total_steps = total_opt = fails = total_collisions = 0
    total_cases = 0

    random.seed(RANDOM_SEED)
    with open(log_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            if task_key == "agent_multi_yesno" or task_key == "agent_yesno_unassigned" or task_key == "agent_com_unassigned":
                writer.writerow(["World", "Pair", "Trial", "Steps", "Optimal", "Failed", "Collisions"])
            else:
                writer.writerow(["World", "Pair", "Trial", "Steps", "Optimal", "Failed"])

        for w in range(NUM_OBSTACLE_WORLDS):
            print(f"\n--- Obstacle World {w+1} ---")
            dummy_positions = {(0, 0), (GRID_SIZE - 1, GRID_SIZE - 1)}
            obstacles = generate_random_obstacles(GRID_SIZE, avoid=dummy_positions)

            if task_key == "agent_multi_yesno" or task_key == "agent_yesno_unassigned" or task_key == "agent_com_unassigned":
                # Multi-agent evaluation
                for pair_idx in range(AGENT_GOAL_PAIRS_PER_WORLD):
                    for t in range(TRIALS_PER_CONFIG):
                        print(f"World {w+1}, Pair {pair_idx+1}, Trial {t+1} — Running multi-agent")
                        steps, optimal, failed, collisions = run_fn(
                            grid_size=GRID_SIZE,
                            obstacles=obstacles,
                            image_path=IMAGE_PATH,
                            max_steps=MAX_STEPS,
                            num_agents=NUM_AGENTS
                        )
                        writer.writerow([w+1, pair_idx+1, t+1, steps, optimal, int(failed), collisions])
                        total_steps += steps
                        total_opt += optimal
                        fails += int(failed)
                        total_collisions += collisions
                        total_cases += 1
            else:
                # Single-agent evaluation
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
                            image_path=IMAGE_PATH,
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
        f"Evaluated {total_cases} trials {'(multi-agent)' if task_key == 'agent_multi_yesno' else ''}",
        f"Average optimal path: {avg_opt:.2f}",
        f"Average steps taken: {avg_steps:.2f}",
        f"Difference: {diff:.2f} ({percent:.2f}%)",
        f"Failures (max steps): {fails}/{total_cases}"
    ]

    if task_key == "agent_multi_yesno":
        summary_lines.append(f"Total collisions: {total_collisions}")
        summary_lines.append("Note: includes collision counts in output CSV.")

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
    parser.add_argument("--image-path", type=str, default=IMAGE_PATH, help="Path to grid image file")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Maximum steps per trial")
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE, help="Grid size (NxN)")
    parser.add_argument("--trials", type=int, default=TRIALS_PER_CONFIG, help="Trials per agent-goal pair")
    parser.add_argument("--num-worlds", type=int, default=NUM_OBSTACLE_WORLDS, help="Number of obstacle worlds")
    parser.add_argument("--agent-goal-pairs", type=int, default=AGENT_GOAL_PAIRS_PER_WORLD, help="Agent-goal pairs per world")
    parser.add_argument("--num-obstacles", type=int, default=NUM_OBSTACLES, help="Number of obstacles per world")
    parser.add_argument("--num-agents", type=int, default=NUM_AGENTS, help="Number of agents (for multi-agent agents)")
    args = parser.parse_args()

    # Override constants with args
    IMAGE_PATH = args.image_path
    MAX_STEPS = args.max_steps
    GRID_SIZE = args.grid_size
    TRIALS_PER_CONFIG = args.trials
    NUM_OBSTACLE_WORLDS = args.num_worlds
    AGENT_GOAL_PAIRS_PER_WORLD = args.agent_goal_pairs
    NUM_OBSTACLES = args.num_obstacles
    NUM_AGENTS = args.num_agents

    selected_agents = {key: mod for (key, mod) in TASKS if key in args.agents}
    for key, module_path in selected_agents.items():
        module = importlib.import_module(module_path)
        run_fn = module.run
        evaluate_random(key, run_fn)
