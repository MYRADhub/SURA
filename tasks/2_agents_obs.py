from core.environment import GridWorld
from core.prompt import build_prompt_first_agent_obs, build_prompt_second_agent_obs
from core.agent import send_image_to_model_openai
from core.plot import plot_grid
import random

def extract_direction(response):
    for d in ['up', 'down', 'left', 'right']:
        if d in response.lower():
            return d
    return None

if __name__ == "__main__":
    grid_size = 6
    image_path = "data/grid.png"

    obstacles = {(1, 1), (2, 3), (4, 2), (3, 4)}
    env = GridWorld(grid_size, obstacles=obstacles)
    env.initialize_agents_goals(num_agents=2)

    agent1_pos, agent2_pos = env.agents
    goal1_pos, goal2_pos = env.goals
    init_a1, init_a2 = agent1_pos, agent2_pos

    step = 0
    done1 = done2 = False
    collisions = 0

    while not (done1 and done2):
        print(f"\n--- Step {step} ---")
        print(f"A1: {agent1_pos} → {goal1_pos}")
        print(f"A2: {agent2_pos} → {goal2_pos}")

        plot_grid(env, image_path=image_path)

        new_a1, new_a2 = agent1_pos, agent2_pos

        if not done1:
            prompt1 = build_prompt_first_agent_obs(agent1_pos, agent2_pos, goal1_pos, env.get_valid_actions(agent1_pos), grid_size, obstacles)
            resp1 = send_image_to_model_openai(image_path, prompt1, temperature=0.0000001)
            dir1 = extract_direction(resp1)
            if dir1:
                new_a1 = env.move_agent(agent1_pos, dir1)

        if not done2:
            prompt2 = build_prompt_second_agent_obs(agent1_pos, agent2_pos, goal2_pos, env.get_valid_actions(agent2_pos), grid_size, obstacles)
            resp2 = send_image_to_model_openai(image_path, prompt2, temperature=0.0000001)
            dir2 = extract_direction(resp2)
            if dir2:
                new_a2 = env.move_agent(agent2_pos, dir2)

        # Prevent collision
        if new_a1 == new_a2:
            collisions += 1
            if random.random() < 0.5:
                new_a2 = agent2_pos
            else:
                new_a1 = agent1_pos

        agent1_pos, agent2_pos = new_a1, new_a2
        env.agents = [agent1_pos, agent2_pos]

        if agent1_pos == goal1_pos:
            done1 = True
        if agent2_pos == goal2_pos:
            done2 = True

        step += 1

    print("\nTask completed!")
    print(f"Steps: {step}, Collisions: {collisions}")
    print(f"Optimal A1: {abs(init_a1[0] - goal1_pos[0]) + abs(init_a1[1] - goal1_pos[1])}")
    print(f"Optimal A2: {abs(init_a2[0] - goal2_pos[0]) + abs(init_a2[1] - goal2_pos[1])}")
