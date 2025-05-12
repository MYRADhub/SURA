from core.plot import plot_grid
from core.agent import send_image_to_model_openai
from core.prompt import build_prompt_single
from core.environment import GridWorld

MAX_STEPS = 30
GRID_SIZE = 6
NUM_RUNS = 20
IMAGE_PATH = "data/grid.png"

def run_single_episode():
    env = GridWorld(GRID_SIZE, obstacles={(1, 1), (3, 3)})
    env.initialize_agents_goals(num_agents=1)

    agent_pos = env.agents[0]
    goal_pos = env.goals[0]
    init_agent_pos = agent_pos
    step = 0

    while agent_pos != goal_pos and step < MAX_STEPS:
        plot_grid(env, image_path=IMAGE_PATH)

        valid_actions = env.get_valid_actions(agent_pos)
        prompt = build_prompt_single(agent_pos, goal_pos, valid_actions, GRID_SIZE)
        response = send_image_to_model_openai(IMAGE_PATH, prompt, temperature=0.0000001)

        direction = next((d for d in ['up', 'down', 'left', 'right'] if d in response), None)
        if not direction:
            break

        new_pos = env.move_agent(agent_pos, direction)
        if new_pos == agent_pos:
            break  # Invalid or blocked
        else:
            agent_pos = new_pos
            env.agents[0] = agent_pos
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
