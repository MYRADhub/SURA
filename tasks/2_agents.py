from rendering.plot_2_agents import plot_grid
from core.agent import send_image_to_model_openai
from core.prompt import build_prompt_first_agent, build_prompt_second_agent
from core.environment import GridWorld
import random

def extract_direction(response):
    for dir_candidate in ['up', 'down', 'left', 'right']:
        if dir_candidate in response.lower():
            return dir_candidate
    return None

if __name__ == "__main__":
    grid_size = 6
    image_path = "data/grid_2_agents.png"
    
    env = GridWorld(grid_size)
    env.initialize_agents_goals(num_agents=2)

    agent1_pos, agent2_pos = env.agents
    goal1_pos, goal2_pos = env.goals
    init_agent1_pos, init_agent2_pos = agent1_pos, agent2_pos

    step = 0
    agent1_done = agent2_done = False
    collision_count = 0

    while not (agent1_done and agent2_done):
        print(f"\n--- Step {step} ---")
        print(f"A1: {agent1_pos} -> {goal1_pos}")
        print(f"A2: {agent2_pos} -> {goal2_pos}")

        plot_grid(env, image_path=image_path)

        new_agent1_pos, new_agent2_pos = agent1_pos, agent2_pos

        if not agent1_done:
            prompt1 = build_prompt_first_agent(agent1_pos, agent2_pos, goal1_pos, env.get_valid_actions(agent1_pos), grid_size)
            response1 = send_image_to_model_openai(image_path, prompt1, temperature=0.0000001)
            direction1 = extract_direction(response1)
            if direction1:
                new_agent1_pos = env.move_agent(agent1_pos, direction1)

        if not agent2_done:
            prompt2 = build_prompt_second_agent(agent1_pos, agent2_pos, goal2_pos, env.get_valid_actions(agent2_pos), grid_size)
            response2 = send_image_to_model_openai(image_path, prompt2, temperature=0.0000001)
            direction2 = extract_direction(response2)
            if direction2:
                new_agent2_pos = env.move_agent(agent2_pos, direction2)

        if new_agent1_pos == new_agent2_pos:
            collision_count += 1
            if random.random() < 0.5:
                new_agent2_pos = agent2_pos
            else:
                new_agent1_pos = agent1_pos

        agent1_pos, agent2_pos = new_agent1_pos, new_agent2_pos
        env.agents = [agent1_pos, agent2_pos]

        if agent1_pos == goal1_pos:
            agent1_done = True
        if agent2_pos == goal2_pos:
            agent2_done = True

        step += 1

    print("\nTask completed!")
    print(f"Steps: {step}, Collisions: {collision_count}")
    print(f"Optimal A1: {abs(init_agent1_pos[0] - goal1_pos[0]) + abs(init_agent1_pos[1] - goal1_pos[1])}")
    print(f"Optimal A2: {abs(init_agent2_pos[0] - goal2_pos[0]) + abs(init_agent2_pos[1] - goal2_pos[1])}")
