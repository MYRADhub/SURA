from core.environment import GridWorld
from core.prompt import build_prompt_single_obs
from core.agent import send_image_to_model_openai
from core.plot import plot_grid

if __name__ == "__main__":
    grid_size = 6
    image_path = "data/grid.png"
    
    obstacles = {(2, 2), (3, 3), (1, 4)}
    env = GridWorld(grid_size, obstacles=obstacles)
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
        prompt = build_prompt_single_obs(agent_pos, goal_pos, valid_actions, grid_size, obstacles)

        response = send_image_to_model_openai(image_path, prompt, temperature=0.0000001)

        print(f"\nPrompt:\n{prompt}\n")
        print(f"LLM response: {response}")

        direction = next((d for d in ['up', 'down', 'left', 'right'] if d in response), None)

        if not direction:
            print("No valid direction, stopping.")
            break

        new_agent_pos = env.move_agent(agent_pos, direction)
        if new_agent_pos == agent_pos:
            print("Move blocked or invalid.")
        else:
            agent_pos = new_agent_pos
            env.agents[0] = agent_pos
            step += 1

    print("\nGoal reached!")
    optimal = abs(init_agent_pos[0] - goal_pos[0]) + abs(init_agent_pos[1] - goal_pos[1])
    print(f"Optimal: {optimal}, Steps taken: {step}")
