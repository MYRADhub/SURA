from core.environment import GridWorld
from core.prompt import build_prompt_single_obs, build_prompt_single_obs_v2
from core.request import send_image_to_model_openai
from core.plot import plot_grid
from core.utils import shortest_path_length

def extract_direction(response):
    for d in ['up', 'down', 'left', 'right']:
        if d == response.lower():
            return d
    return None

def run(
    obstacles={(2, 2), (3, 3), (1, 4)},
    grid_size=6,
    image_path="data/grid.png",
    max_steps=30,
    agent_start: tuple[int, int] = None,
    goal_pos: tuple[int, int] = None
):
    env = GridWorld(grid_size, obstacles=obstacles)
    if agent_start and goal_pos:
        env.initialize_agents_goals_custom(agents=[agent_start], goals=[goal_pos])
    else:
        env.initialize_agents_goals(num_agents=1)

    agent_pos = env.agents[0]
    goal_pos = env.goals[0]
    init_agent_pos = agent_pos

    step = 0
    memory = []  # Store tuples of (x0, y0, direction, x1, y1)
    visits = {}  # Track number of visits to each cell
    failed = False

    while agent_pos != goal_pos and step < max_steps:
        plot_grid(env, image_path=image_path)

        # Update visits count
        visits[agent_pos] = visits.get(agent_pos, 0) + 1

        valid_actions = env.get_valid_actions(agent_pos)
        prompt = build_prompt_single_obs_v2(
            agent_pos, goal_pos, valid_actions, grid_size, obstacles, memory, visits
        )
        # prompt = build_prompt_single_obs(
        #     agent_pos, goal_pos, valid_actions, grid_size, obstacles
        # )
        print(f"Valid actions: {valid_actions}")
        # print(f"Prompt: {prompt}")
        response = send_image_to_model_openai(image_path, prompt, temperature=0.0000001)
        print(f"Response: {response}")

        direction = extract_direction(response)
        if not direction:
            break

        new_agent_pos = env.move_agent(agent_pos, direction)
        if new_agent_pos == agent_pos:
            failed = True
            break

        # Update memory with the action and new position, keep only last 5 actions
        memory.append((agent_pos[0], agent_pos[1], direction, new_agent_pos[0], new_agent_pos[1]))
        if len(memory) > 5:
            memory = memory[-5:]

        agent_pos = new_agent_pos
        env.agents[0] = agent_pos
        step += 1

    optimal = shortest_path_length(init_agent_pos, goal_pos, env)
    failed = step >= max_steps or failed
    return step, optimal, failed

if __name__ == "__main__":
    steps, optimal, failed = run()
    print(f"\n✅ Goal reached!\nOptimal: {optimal}, Steps taken: {steps}, Failed: {failed}")
