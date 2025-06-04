from core.environment import GridWorld
from core.prompt import build_yesno_code_prompt_single
from core.agent import send_image_to_model_openai_logprobs
from core.plot import plot_grid
from core.utils import shortest_path_length
import time

def extract_yes_logprob(logprobs):
    """Extract logprob for the token 'yes' from OpenAI response"""
    if not logprobs:
        return float('-inf')
    for item in logprobs[0].top_logprobs:
        if item.token.strip().lower() == "yes":
            return item.logprob
    return float('-inf')

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
    memory = []  # list of (r0, c0, direction, r1, c1)
    visits = {}
    failed = False

    while agent_pos != goal_pos and step < max_steps:
        plot_grid(env, image_path=image_path)

        visits[agent_pos] = visits.get(agent_pos, 0) + 1
        valid_actions = env.get_valid_actions(agent_pos)
        print(f"\n--- Step {step} ---")
        print(f"Agent: {agent_pos} → Goal: {goal_pos}")
        print(f"Valid actions: {valid_actions}")

        # Ask yes/no for each valid direction
        action_scores = {}
        for direction in valid_actions:
            prompt = build_yesno_code_prompt_single(
                agent_pos, goal_pos, grid_size, obstacles, direction, memory, visits
            )
            time.sleep(0.5)  # Avoid rate limiting
            sentence, logprobs = send_image_to_model_openai_logprobs(image_path, prompt, temperature=0.0000001)
            logprob_yes = extract_yes_logprob(logprobs)
            action_scores[direction] = logprob_yes
            print(f"{direction.upper():5} → logprob(yes): {logprob_yes:.3f}")
        time.sleep(1)

        if not action_scores:
            print("No direction could be chosen.")
            failed = True
            break

        # Choose direction with highest logprob for "yes"
        direction = max(action_scores, key=action_scores.get)
        new_agent_pos = env.move_agent(agent_pos, direction)
        print(f"Selected: {direction}, moving to {new_agent_pos}")

        if new_agent_pos == agent_pos:
            print("Move blocked. Exiting.")
            failed = True
            break

        # Track memory
        memory.append((agent_pos[0], agent_pos[1], direction, new_agent_pos[0], new_agent_pos[1]))
        if len(memory) > 5:
            memory = memory[-5:]

        agent_pos = new_agent_pos
        env.agents[0] = agent_pos
        step += 1

    optimal = shortest_path_length(init_agent_pos, goal_pos, env)
    failed = failed or step >= max_steps
    return step, optimal, failed

if __name__ == "__main__":
    steps, optimal, failed = run()
    print(f"\n✅ Finished")
    print(f"Optimal path length: {optimal}")
    print(f"Steps taken        : {steps}")
    print(f"Failure            : {failed}")
