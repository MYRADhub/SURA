import random
from plot import plot_grid
from agent import send_image_to_model_openai, send_image_to_model_ollama
from utils import get_valid_actions, move_agent, build_prompt_single

def get_random_positions(grid_size):
    positions = random.sample(range(grid_size * grid_size), 2)
    agent_pos = divmod(positions[0], grid_size)
    goal_pos = divmod(positions[1], grid_size)
    return agent_pos, goal_pos

if __name__ == "__main__":
    grid_size = 6
    agent_pos, goal_pos = get_random_positions(grid_size)
    init_agent_pos = agent_pos

    step = 0
    image_path = "grid.png"
    while agent_pos != goal_pos:
        print(f"\n--- Step {step} ---")
        print(f"Agent: {agent_pos}, Goal: {goal_pos}")

        plot_grid(grid_size, agent_pos, goal_pos, image_path=image_path)
        print(f"Grid saved as '{image_path}'")

        valid_actions = get_valid_actions(agent_pos, grid_size)
        prompt = build_prompt_single(agent_pos, goal_pos, valid_actions, grid_size)

        print("Sending image to the model...")

        response = send_image_to_model_openai(image_path, prompt, temperature=0.0000001)
        print(f"\n\nPrompt: {prompt}\n\n")
        print(f"LLM response: {response}\n\n")

        # Extract direction (simple keyword check)
        for dir_candidate in ['up', 'down', 'left', 'right']:
            if dir_candidate in response:
                direction = dir_candidate
                break
        else:
            print("No valid direction found, stopping.")
            break

        new_agent_pos = move_agent(agent_pos, direction, grid_size)
        if new_agent_pos == agent_pos:
            print("Move would go out of bounds, ignoring action.")
        else:
            agent_pos = new_agent_pos
            step += 1

    print("\nGoal reached!")
    # The optimal path length:
    optimal_path_length = abs(init_agent_pos[0] - goal_pos[0]) + abs(init_agent_pos[1] - goal_pos[1])
    print(f"Optimal path length: {optimal_path_length}")
    print(f"Total steps taken: {step}")
