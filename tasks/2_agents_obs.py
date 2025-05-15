from core.environment import GridWorld
from core.prompt import build_prompt_first_agent_obs, build_prompt_second_agent_obs, build_prompt_single_obs
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
    deleted1 = deleted2 = False
    collisions = 0

    while not (done1 and done2):
        print(f"\n--- Step {step} ---")
        print(f"A1: {agent1_pos} → {goal1_pos}")
        print(f"A2: {agent2_pos} → {goal2_pos}")

        plot_grid(env, image_path=image_path)

        new_a1, new_a2 = agent1_pos, agent2_pos

        # If both agents are present
        if not deleted1 and not deleted2:
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

        # Only agent 1 remains
        elif not deleted1 and deleted2:
            if not done1:
                prompt1 = build_prompt_single_obs(agent1_pos, goal1_pos, env.get_valid_actions(agent1_pos), grid_size, obstacles)
                resp1 = send_image_to_model_openai(image_path, prompt1, temperature=0.0000001)
                dir1 = extract_direction(resp1)
                if dir1:
                    new_a1 = env.move_agent(agent1_pos, dir1)
            new_a2 = None  # Agent 2 is deleted

        # Only agent 2 remains
        elif deleted1 and not deleted2:
            if not done2:
                prompt2 = build_prompt_single_obs(agent2_pos, goal2_pos, env.get_valid_actions(agent2_pos), grid_size, obstacles)
                resp2 = send_image_to_model_openai(image_path, prompt2, temperature=0.0000001)
                dir2 = extract_direction(resp2)
                if dir2:
                    new_a2 = env.move_agent(agent2_pos, dir2)
            new_a1 = None  # Agent 1 is deleted

        # Update agent positions
        agent1_pos = new_a1 if new_a1 is not None else agent1_pos
        agent2_pos = new_a2 if new_a2 is not None else agent2_pos
        env.agents = [pos for pos in [agent1_pos, agent2_pos] if pos is not None]

        if not deleted1 and agent1_pos == goal1_pos:
            done1 = True
        if not deleted2 and agent2_pos == goal2_pos:
            done2 = True

        if done1 and not deleted1:
            deleted1 = True
            env.remove_agent(0)
            env.remove_goal(0)
            agent1_pos = None
        if done2 and not deleted2:
            idx = 1 if not deleted1 else 0
            deleted2 = True
            env.remove_agent(idx)
            env.remove_goal(idx)
            agent2_pos = None
        if deleted1 and deleted2:
            break

        step += 1

    print("\nTask completed!")
    print(f"Steps: {step}, Collisions: {collisions}")
    print(f"Optimal A1: {abs(init_a1[0] - goal1_pos[0]) + abs(init_a1[1] - goal1_pos[1])}")
    print(f"Optimal A2: {abs(init_a2[0] - goal2_pos[0]) + abs(init_a2[1] - goal2_pos[1])}")
