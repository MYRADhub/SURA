from core.environment import GridWorld
from core.prompt import build_yesno_prompt_multiagent
from core.request import send_image_to_model_openai_logprobs
from core.plot import plot_grid
from core.utils import shortest_path_length
import time

def extract_yes_logprob(logprobs):
    if not logprobs:
        return float('-inf')
    for item in logprobs[0].top_logprobs:
        if item.token.strip() == "YES":
            return item.logprob
    return float('-inf')

def run(
    image_path="data/grid.png",
    log_path="data/agent_step_logs.csv",
    max_steps=30,
    config_path=None,
    obstacles={(2, 2), (3, 3), (4, 1)},
    grid_size=6,
    num_agents=3,
    agent_starts: list[tuple[int, int]] = None,
    goal_positions: list[tuple[int, int]] = None
):
    if config_path:
        env = GridWorld(config_path)
        grid_size = env.size
        num_agents = len(env.agents)
    else:
        env = GridWorld(grid_size, obstacles=obstacles)
        if agent_starts and goal_positions:
            env.initialize_agents_goals_custom(agents=agent_starts, goals=goal_positions)
        else:
            env.initialize_agents_goals(num_agents=num_agents)

    init_positions = env.agents[:]
    goal_positions = env.goals[:]
    agent_positions = env.agents[:]
    agent_ids = list(range(1, num_agents + 1))
    active = [True] * num_agents
    visits = [{} for _ in range(num_agents)]
    memories = [[] for _ in range(num_agents)]
    step = 0
    collisions = 0
    optimal_lengths = [
        shortest_path_length(init_positions[i], goal_positions[i], env)
        for i in range(num_agents)
    ]

    while any(active) and step < max_steps:
        plot_grid(env, image_path=image_path)
        proposals = [pos for pos in agent_positions]

        for i in range(num_agents):
            if not active[i] or agent_positions[i] is None:
                continue
            visits[i][agent_positions[i]] = visits[i].get(agent_positions[i], 0) + 1
            valid = env.get_valid_actions(agent_positions[i])
            scores = {}
            for d in valid:
                other_infos = [
                    (agent_ids[j], agent_positions[j])
                    for j in range(num_agents)
                    if j != i and agent_positions[j] is not None
                ]
                prompt = build_yesno_prompt_multiagent(
                    agent_id=agent_ids[i],
                    agent_pos=agent_positions[i],
                    goal_pos=goal_positions[i],
                    other_agents=other_infos,
                    grid_size=grid_size,
                    obstacles=obstacles,
                    direction=d,
                    memory=memories[i],
                    visits=visits[i]
                )
                print(f"\nAgent {agent_ids[i]} at position {agent_positions[i]}, asking about direction '{d}'")
                print(f"Prompt: {prompt[:200]}...")
                time.sleep(0.5)
                _, logprobs = send_image_to_model_openai_logprobs(image_path, prompt, temperature=0.0000001)
                scores[d] = extract_yes_logprob(logprobs)
            if scores:
                best = max(scores, key=scores.get)
                proposals[i] = env.move_agent(agent_positions[i], best)
                memories[i].append((agent_positions[i][0], agent_positions[i][1], best, proposals[i][0], proposals[i][1]))
                if len(memories[i]) > 5:
                    memories[i] = memories[i][-5:]
            else:
                proposals[i] = agent_positions[i]

        # Collision resolution
        new_positions = proposals[:]
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if (
                    agent_positions[i] is not None and
                    agent_positions[j] is not None and
                    new_positions[i] == new_positions[j]
                ):
                    collisions += 1
                    # Prioritize agent with lower index
                    new_positions[j] = agent_positions[j]

        agent_positions = new_positions
        env.agents = agent_positions[:]

        # Goal checking and cleanup
        to_remove = []
        for i in range(num_agents):
            if not active[i] or agent_positions[i] is None:
                continue
            if agent_positions[i] == goal_positions[i]:
                active[i] = False
                to_remove.append(i)

        for i in to_remove:
            agent_positions[i] = None
            env.agents[i] = None
            goal_positions[i] = None
            env.goals[i] = None

        step += 1
    
    failed = step >= max_steps
    return step, max(optimal_lengths), failed, collisions

if __name__ == "__main__":
    steps, optimal, failed, collisions = run()
    print(f"\n✅ Task completed!\nOptimal: {optimal}, Steps: {steps}, Failed: {failed}, Collisions: {collisions}")
