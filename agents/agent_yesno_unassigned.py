from core.environment import GridWorld
from core.prompt import build_yesno_prompt_unassigned_goals
from core.agent import send_image_to_model_openai_logprobs
from core.plot import plot_grid_unassigned
from core.utils import shortest_path_length
import time

def extract_yes_logprob(logprobs):
    if not logprobs:
        return float('-inf')
    for item in logprobs[0].top_logprobs:
        if item.token.strip().lower() == "yes":
            return item.logprob
    return float('-inf')

def run(obstacles={(2, 2), (3, 3), (4, 1)}, grid_size=6, image_path="data/grid.png", max_steps=30, num_agents=3):
    env = GridWorld(grid_size, obstacles=obstacles)
    env.initialize_agents_goals(num_agents=num_agents)

    agent_positions = env.agents[:]
    agent_ids = list(range(1, num_agents + 1))
    active = [True] * num_agents
    visits = [{} for _ in range(num_agents)]
    memories = [[] for _ in range(num_agents)]
    step = 0
    collisions = 0
    # Compute optimal steps (hypothetical, assuming original assignment)
    total_opt = 0
    for start in env.agents:
        if start is None:
            continue
        dists = [shortest_path_length(start, g, env) for g in env.goals if g is not None]
        total_opt += min(dists) if dists else 0

    print(f"Initial agent positions: {agent_positions}")
    print(f"Goal positions: {env.goals}")
    print(f"Obstacles: {obstacles}")

    while any(active) and step < max_steps:
        print(f"\n--- Step {step} ---")
        plot_grid_unassigned(env, image_path=image_path)
        proposals = agent_positions[:]

        for i in range(num_agents):
            if not active[i] or agent_positions[i] is None:
                continue

            print(f"\nAgent {agent_ids[i]} at position {agent_positions[i]}")
            visits[i][agent_positions[i]] = visits[i].get(agent_positions[i], 0) + 1
            valid = env.get_valid_actions(agent_positions[i])
            print(f"Valid moves for Agent {agent_ids[i]}: {valid}")
            scores = {}

            for d in valid:
                other_infos = [
                    (agent_ids[j], agent_positions[j])
                    for j in range(num_agents)
                    if j != i and agent_positions[j] is not None
                ]
                prompt = build_yesno_prompt_unassigned_goals(
                    agent_id=agent_ids[i],
                    agent_pos=agent_positions[i],
                    goal_positions=env.goals,
                    other_agents=other_infos,
                    grid_size=grid_size,
                    obstacles=obstacles,
                    direction=d,
                    memory=memories[i],
                    visits=visits[i]
                )
                print(f"Agent {agent_ids[i]} prompt for direction {d}: {prompt[:200]}...")
                time.sleep(0.5)
                _, logprobs = send_image_to_model_openai_logprobs(image_path, prompt, temperature=0.0000001)
                score = extract_yes_logprob(logprobs)
                scores[d] = score
                print(f"Agent {agent_ids[i]} logprob for direction {d}: {score}")

            if scores:
                best = max(scores, key=scores.get)
                print(f"Agent {agent_ids[i]} chooses direction {best} with logprob {scores[best]}")
                proposals[i] = env.move_agent(agent_positions[i], best)
                print(f"Agent {agent_ids[i]} moves from {agent_positions[i]} to {proposals[i]}")
                memories[i].append((agent_positions[i][0], agent_positions[i][1], best, proposals[i][0], proposals[i][1]))
                if len(memories[i]) > 5:
                    memories[i] = memories[i][-5:]
            else:
                print(f"Agent {agent_ids[i]} has no valid moves and stays at {agent_positions[i]}")
                proposals[i] = agent_positions[i]

        # Collision resolution
        new_positions = proposals[:]
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if new_positions[i] == new_positions[j] and agent_positions[i] is not None and agent_positions[j] is not None:
                    collisions += 1
                    print(f"Collision detected between Agent {agent_ids[i]} and Agent {agent_ids[j]} at {new_positions[i]}")
                    new_positions[j] = agent_positions[j]  # loser stays

        agent_positions = new_positions
        print(f"Agent positions after moves: {agent_positions}")
        env.agents = agent_positions[:]

        # Goal claiming logic
        to_remove = []
        for i in range(num_agents):
            if not active[i] or agent_positions[i] is None:
                continue
            if agent_positions[i] in env.goals:
                print(f"Agent {agent_ids[i]} reached a goal at {agent_positions[i]}")
                active[i] = False
                to_remove.append(i)

        for i in to_remove:
            agent_positions[i] = None
            env.agents[i] = None
            env.goals[i] = None

        print(f"Active agents: {[agent_ids[i] for i in range(num_agents) if active[i] and agent_positions[i] is not None]}")
        print(f"Remaining goals: {env.goals}")

        step += 1

    failed = step >= max_steps
    print(f"\nRun finished in {step} steps. Collisions: {collisions}. Failed: {failed}")
    return step, total_opt, failed, collisions

if __name__ == "__main__":
    steps, optimal, failed, collisions = run()
    print(f"\n✅ Task completed!")
    print(f"Optimal: {optimal}, Steps: {steps}, Failed: {failed}, Collisions: {collisions}")
