from core.plot import plot_grid
from core.agent import send_image_to_model_openai
from core.prompt import build_prompt_single
from core.environment import GridWorld

if __name__ == "__main__":
    grid_size = 6
    image_path = "data/grid.png"
    
    env = GridWorld(grid_size)  
    env.initialize_agents_goals(num_agents=1)

    agent_pos = env.agents[0]
    goal_pos = env.goals[0]
    init_agent_pos = agent_pos

    step = 0
    while agent_pos != goal_pos:
        print(f"\n--- Step {step} ---")
        print(f"Agent: {agent_pos}, Goal: {goal_pos}")

        plot_grid(env, image_path=image_path)

        valid_actions = env.get_valid_actions(agent_pos)
        prompt = build_prompt_single(agent_pos, goal_pos, valid_actions, grid_size)

        print("Sending image to the model...")
        response = send_image_to_model_openai(image_path, prompt, temperature=0.0000001)

        print(f"\nPrompt:\n{prompt}\n")
        print(f"LLM response: {response}")

        direction = next((d for d in ['up', 'down', 'left', 'right'] if d in response), None)
        if not direction:
            print("No valid direction found, stopping.")
            break

        new_agent_pos = env.move_agent(agent_pos, direction)
        if new_agent_pos == agent_pos:
            print("Move would go out of bounds or into obstacle, ignoring.")
        else:
            agent_pos = new_agent_pos
            env.agents[0] = agent_pos
            step += 1

    print("\nGoal reached!")
    optimal_path_length = abs(init_agent_pos[0] - goal_pos[0]) + abs(init_agent_pos[1] - goal_pos[1])
    print(f"Optimal path length: {optimal_path_length}")
    print(f"Total steps taken: {step}")
