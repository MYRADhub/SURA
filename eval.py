import random
from plot import plot_grid
from agent import send_image_to_model_openai
from utils import get_valid_actions, move_agent, build_prompt_single

MAX_STEPS = 30
GRID_SIZE = 6
NUM_RUNS = 20
IMAGE_PATH = "grid.png"

def get_random_positions(grid_size):
    positions = random.sample(range(grid_size * grid_size), 2)
    agent_pos = divmod(positions[0], grid_size)
    goal_pos = divmod(positions[1], grid_size)
    return agent_pos, goal_pos

def run_single_episode():
    agent_pos, goal_pos = get_random_positions(GRID_SIZE)
    init_agent_pos = agent_pos
    step = 0

    while agent_pos != goal_pos and step < MAX_STEPS:
        plot_grid(GRID_SIZE, agent_pos, goal_pos, image_path=IMAGE_PATH)

        valid_actions = get_valid_actions(agent_pos, GRID_SIZE)
        prompt = build_prompt_single(agent_pos, goal_pos, valid_actions, GRID_SIZE)
        response = send_image_to_model_openai(
            IMAGE_PATH, prompt, temperature=0.0000001
        )

        # Extract direction
        direction = None
        for dir_candidate in ['up', 'down', 'left', 'right']:
            if dir_candidate in response:
                direction = dir_candidate
                break

        if not direction:
            break  # Invalid response

        new_pos = move_agent(agent_pos, direction, GRID_SIZE)
        if new_pos == agent_pos:
            # Invalid move (would go out of bounds)
            break
        else:
            agent_pos = new_pos
            step += 1

    optimal_path = abs(init_agent_pos[0] - goal_pos[0]) + abs(init_agent_pos[1] - goal_pos[1])
    return step, optimal_path, (step >= MAX_STEPS)

def evaluate_agent():
    total_steps = 0
    total_optimal = 0
    failures = 0

    for i in range(NUM_RUNS):
        print(f"Running episode {i+1}/{NUM_RUNS}...")
        steps, optimal, failed = run_single_episode()
        total_steps += steps
        total_optimal += optimal
        if failed:
            failures += 1

    avg_steps = total_steps / NUM_RUNS
    avg_optimal = total_optimal / NUM_RUNS
    avg_diff = avg_steps - avg_optimal
    percent_diff = (avg_diff / avg_optimal) * 100 if avg_optimal > 0 else 0

    print("\n=== Evaluation Summary ===")
    print(f"Episodes run: {NUM_RUNS}")
    print(f"Average optimal path length: {avg_optimal:.2f}")
    print(f"Average steps taken: {avg_steps:.2f}")
    print(f"Average difference: {avg_diff:.2f}")
    print(f"Percent difference: {percent_diff:.2f}%")
    print(f"Failures (max steps reached): {failures}/{NUM_RUNS}")

if __name__ == "__main__":
    evaluate_agent()
