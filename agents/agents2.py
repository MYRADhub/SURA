from core.plot import plot_grid
from core.request import send_image_to_model_openai
from core.prompt import build_prompt_first_agent, build_prompt_second_agent
from core.environment import GridWorld
from core.utils import shortest_path_length
import random

def extract_direction(response):
    for d in ['up', 'down', 'left', 'right']:
        if d in response.lower():
            return d
    return None

def run(grid_size=6, image_path="data/grid.png", max_steps=30):
    env = GridWorld(grid_size)
    env.initialize_agents_goals(num_agents=2)

    agent1_pos, agent2_pos = env.agents
    goal1_pos, goal2_pos = env.goals
    init1, init2 = agent1_pos, agent2_pos

    step = 0
    done1 = done2 = False
    collisions = 0

    while not (done1 and done2) and step < max_steps:
        plot_grid(env, image_path=image_path)
        new1, new2 = agent1_pos, agent2_pos

        if not done1:
            valid1 = env.get_valid_actions(agent1_pos)
            prompt1 = build_prompt_first_agent(agent1_pos, agent2_pos, goal1_pos, valid1, grid_size)
            resp1 = send_image_to_model_openai(image_path, prompt1, temperature=0.0000001)
            dir1 = extract_direction(resp1)
            if dir1:
                new1 = env.move_agent(agent1_pos, dir1)

        if not done2:
            valid2 = env.get_valid_actions(agent2_pos)
            prompt2 = build_prompt_second_agent(agent1_pos, agent2_pos, goal2_pos, valid2, grid_size)
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

        agent1_pos, agent2_pos = new1, new2
        env.agents = [agent1_pos, agent2_pos]

        done1 = agent1_pos == goal1_pos
        done2 = agent2_pos == goal2_pos
        step += 1

    opt1 = shortest_path_length(init1, goal1_pos, env)
    opt2 = shortest_path_length(init2, goal2_pos, env)
    failed = step >= max_steps
    return step, max(opt1, opt2), failed, collisions

if __name__ == "__main__":
    steps, optimal, failed, collisions = run()
    print(f"\n✅ Task completed!\nOptimal: {optimal}, Steps: {steps}, Failed: {failed}, Collisions: {collisions}")
