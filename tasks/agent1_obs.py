from core.environment import GridWorld
from core.prompt import build_prompt_single_obs
from core.agent import send_image_to_model_openai
from core.plot import plot_grid
from core.utils import shortest_path_length

def extract_direction(response):
    for d in ['up', 'down', 'left', 'right']:
        if d in response.lower():
            return d
    return None

def run(obstacles={(2, 2), (3, 3), (1, 4)}, grid_size=6, image_path="data/grid.png", max_steps=30):
    env = GridWorld(grid_size, obstacles=obstacles)
    env.initialize_agents_goals(num_agents=1)

    agent_pos = env.agents[0]
    goal_pos = env.goals[0]
    init_agent_pos = agent_pos

    step = 0
    while agent_pos != goal_pos and step < max_steps:
        plot_grid(env, image_path=image_path)

        valid_actions = env.get_valid_actions(agent_pos)
        prompt = build_prompt_single_obs(agent_pos, goal_pos, valid_actions, grid_size, obstacles)
        response = send_image_to_model_openai(image_path, prompt, temperature=0.0000001)

        direction = extract_direction(response)
        if not direction:
            break

        new_agent_pos = env.move_agent(agent_pos, direction)
        if new_agent_pos == agent_pos:
            break

        agent_pos = new_agent_pos
        env.agents[0] = agent_pos
        step += 1

    optimal = shortest_path_length(init_agent_pos, goal_pos, env)
    failed = step >= max_steps
    return step, optimal, failed

if __name__ == "__main__":
    steps, optimal, failed = run()
    print(f"\n✅ Goal reached!\nOptimal: {optimal}, Steps taken: {steps}, Failed: {failed}")
