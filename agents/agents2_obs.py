from core.environment import GridWorld
from core.prompt import build_prompt_first_agent_obs, build_prompt_second_agent_obs, build_prompt_single_obs
from core.request import send_image_to_model_openai
from core.plot import plot_grid
from core.utils import shortest_path_length
import random

def extract_direction(response):
    for d in ['up', 'down', 'left', 'right']:
        if d in response.lower():
            return d
    return None

def run(obstacles={(1, 1), (2, 3), (4, 2), (3, 4)}, grid_size=6, image_path="data/grid.png", max_steps=30):
    env = GridWorld(grid_size, obstacles=obstacles)
    env.initialize_agents_goals(num_agents=2)

    agent1_pos, agent2_pos = env.agents
    goal1_pos, goal2_pos = env.goals
    init1, init2 = agent1_pos, agent2_pos

    step = 0
    done1 = done2 = False
    deleted1 = deleted2 = False
    collisions = 0

    while not (done1 and done2) and step < max_steps:
        plot_grid(env, image_path=image_path)
        new1, new2 = agent1_pos, agent2_pos

        if not deleted1 and not deleted2:
            if not done1:
                prompt1 = build_prompt_first_agent_obs(agent1_pos, agent2_pos, goal1_pos, env.get_valid_actions(agent1_pos), grid_size, obstacles)
                resp1 = send_image_to_model_openai(image_path, prompt1, temperature=0.0000001)
                dir1 = extract_direction(resp1)
                if dir1:
                    new1 = env.move_agent(agent1_pos, dir1)

            if not done2:
                prompt2 = build_prompt_second_agent_obs(agent1_pos, agent2_pos, goal2_pos, env.get_valid_actions(agent2_pos), grid_size, obstacles)
                resp2 = send_image_to_model_openai(image_path, prompt2, temperature=0.0000001)
                dir2 = extract_direction(resp2)
                if dir2:
                    new2 = env.move_agent(agent2_pos, dir2)

            if new1 == new2:
                collisions += 1
                if random.random() < 0.5:
                    new2 = agent2_pos
                else:
                    new1 = agent1_pos

        elif not deleted1 and deleted2:
            if not done1:
                prompt1 = build_prompt_single_obs(agent1_pos, goal1_pos, env.get_valid_actions(agent1_pos), grid_size, obstacles)
                resp1 = send_image_to_model_openai(image_path, prompt1, temperature=0.0000001)
                dir1 = extract_direction(resp1)
                if dir1:
                    new1 = env.move_agent(agent1_pos, dir1)
            new2 = None

        elif deleted1 and not deleted2:
            if not done2:
                prompt2 = build_prompt_single_obs(agent2_pos, goal2_pos, env.get_valid_actions(agent2_pos), grid_size, obstacles)
                resp2 = send_image_to_model_openai(image_path, prompt2, temperature=0.0000001)
                dir2 = extract_direction(resp2)
                if dir2:
                    new2 = env.move_agent(agent2_pos, dir2)
            new1 = None

        agent1_pos = new1 if new1 is not None else agent1_pos
        agent2_pos = new2 if new2 is not None else agent2_pos
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

    opt1 = shortest_path_length(init1, goal1_pos, env)
    opt2 = shortest_path_length(init2, goal2_pos, env)
    failed = step >= max_steps
    return step, max(opt1, opt2), failed, collisions

if __name__ == "__main__":
    steps, optimal, failed, collisions = run()
    print(f"\n✅ Task completed!\nOptimal: {optimal}, Steps: {steps}, Failed: {failed}, Collisions: {collisions}")
