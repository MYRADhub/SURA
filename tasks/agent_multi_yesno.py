from core.environment import GridWorld
from core.prompt import build_yesno_prompt_multiagent
from core.agent import send_image_to_model_openai_logprobs
from core.plot import plot_grid
from core.utils import shortest_path_length
import time

def extract_yes_logprob(logprobs):
    if not logprobs:
        return float('-inf')
    for item in logprobs[0].top_logprobs:
        if item.token.strip().lower() == "yes":
            return item.logprob
    return float('-inf')

def run(obstacles={(1, 1), (2, 3), (4, 2), (3, 4)}, grid_size=6, image_path="data/grid.png", max_steps=30, num_agents=3):
    env = GridWorld(grid_size, obstacles=obstacles)
    env.initialize_agents_goals(num_agents=num_agents)

    init_positions = env.agents[:]
    goal_positions = env.goals[:]
    agent_positions = env.agents[:]
    active = [True] * num_agents
    deleted = [False] * num_agents
    visits = [{} for _ in range(num_agents)]
    memories = [[] for _ in range(num_agents)]
    step = 0
    collisions = 0

    while any(active) and step < max_steps:
        plot_grid(env, image_path=image_path)
        proposals = [pos for pos in agent_positions]

        for i in range(num_agents):
            if not active[i] or deleted[i]:
                continue
            visits[i][agent_positions[i]] = visits[i].get(agent_positions[i], 0) + 1
            valid = env.get_valid_actions(agent_positions[i])
            scores = {}
            for d in valid:
                other_infos = [
                    (j + 1, agent_positions[j])
                    for j in range(num_agents)
                    if j != i and not deleted[j]
                ]
                prompt = build_yesno_prompt_multiagent(
                    agent_id=i + 1,
                    agent_pos=agent_positions[i],
                    goal_pos=goal_positions[i],
                    other_agents=other_infos,
                    grid_size=grid_size,
                    obstacles=obstacles,
                    direction=d,
                    memory=memories[i],
                    visits=visits[i]
                )
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
                if new_positions[i] == new_positions[j] and not deleted[i] and not deleted[j]:
                    collisions += 1
                    # Prioritize agent with lower index
                    new_positions[j] = agent_positions[j]

        agent_positions = new_positions
        env.agents = [pos for i, pos in enumerate(agent_positions) if not deleted[i]]

        # Goal checking and cleanup
        to_remove = []
        for i in range(num_agents):
            if not active[i] or deleted[i]:
                continue
            if agent_positions[i] == goal_positions[i]:
                active[i] = False
                deleted[i] = True
                to_remove.append(i)

        # Remove agents and goals by position to avoid index errors
        for i in sorted(to_remove, reverse=True):
            agent_pos = agent_positions[i]
            goal_pos = goal_positions[i]
            if agent_pos in env.agents:
                env.agents.remove(agent_pos)
            if goal_pos in env.goals:
                env.goals.remove(goal_pos)
            agent_positions[i] = None


        step += 1

    optimal_lengths = [
        shortest_path_length(init_positions[i], goal_positions[i], env)
        for i in range(num_agents)
    ]
    failed = step >= max_steps
    return step, max(optimal_lengths), failed, collisions

if __name__ == "__main__":
    steps, optimal, failed, collisions = run()
    print(f"\n✅ Task completed!\nOptimal: {optimal}, Steps: {steps}, Failed: {failed}, Collisions: {collisions}")
