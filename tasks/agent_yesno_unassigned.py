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
    goal_positions = env.goals[:]
    init_positions = env.agents[:]
    active = [True] * num_agents
    deleted = [False] * num_agents
    visits = [{} for _ in range(num_agents)]
    memories = [[] for _ in range(num_agents)]
    step = 0
    collisions = 0

    print(f"Initial agent positions: {agent_positions}")
    print(f"Goal positions: {goal_positions}")
    print(f"Obstacles: {obstacles}")

    while any(active) and step < max_steps:
        print(f"\n--- Step {step} ---")
        plot_grid_unassigned(env, image_path=image_path)
        proposals = agent_positions[:]

        for i in range(num_agents):
            if not active[i] or deleted[i]:
                continue

            print(f"\nAgent {i+1} at position {agent_positions[i]}")
            visits[i][agent_positions[i]] = visits[i].get(agent_positions[i], 0) + 1
            valid = env.get_valid_actions(agent_positions[i])
            print(f"Valid moves for Agent {i+1}: {valid}")
            scores = {}

            for d in valid:
                other_infos = [
                    (j + 1, agent_positions[j])
                    for j in range(num_agents)
                    if j != i and not deleted[j]
                ]
                prompt = build_yesno_prompt_unassigned_goals(
                    agent_id=i + 1,
                    agent_pos=agent_positions[i],
                    goal_positions=goal_positions,
                    other_agents=other_infos,
                    grid_size=grid_size,
                    obstacles=obstacles,
                    direction=d,
                    memory=memories[i],
                    visits=visits[i]
                )
                print(f"Agent {i+1} prompt for direction {d}: {prompt[:200]}...")  # Print first 200 chars
                time.sleep(0.5)
                _, logprobs = send_image_to_model_openai_logprobs(image_path, prompt, temperature=0.0000001)
                score = extract_yes_logprob(logprobs)
                scores[d] = score
                print(f"Agent {i+1} logprob for direction {d}: {score}")

            if scores:
                best = max(scores, key=scores.get)
                print(f"Agent {i+1} chooses direction {best} with logprob {scores[best]}")
                proposals[i] = env.move_agent(agent_positions[i], best)
                print(f"Agent {i+1} moves from {agent_positions[i]} to {proposals[i]}")
                memories[i].append((agent_positions[i][0], agent_positions[i][1], best, proposals[i][0], proposals[i][1]))
                if len(memories[i]) > 5:
                    memories[i] = memories[i][-5:]
            else:
                print(f"Agent {i+1} has no valid moves and stays at {agent_positions[i]}")
                proposals[i] = agent_positions[i]

        # Collision resolution
        new_positions = proposals[:]
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if new_positions[i] == new_positions[j] and not deleted[i] and not deleted[j]:
                    collisions += 1
                    print(f"Collision detected between Agent {i+1} and Agent {j+1} at {new_positions[i]}")
                    new_positions[j] = agent_positions[j]  # loser stays

        agent_positions = new_positions
        print(f"Agent positions after moves: {agent_positions}")
        env.agents = [pos for i, pos in enumerate(agent_positions) if not deleted[i]]

        # Goal claiming logic
        to_remove = []
        for i in range(num_agents):
            if not active[i] or deleted[i]:
                continue
            if agent_positions[i] in goal_positions:
                print(f"Agent {i+1} reached a goal at {agent_positions[i]}")
                active[i] = False
                deleted[i] = True
                to_remove.append(i)

        for i in sorted(to_remove, reverse=True):
            agent_pos = agent_positions[i]
            if agent_pos in env.agents:
                env.agents.remove(agent_pos)
            if agent_pos in env.goals:
                env.goals.remove(agent_pos)
            agent_positions[i] = None

        print(f"Active agents: {[i+1 for i, a in enumerate(active) if a and not deleted[i]]}")
        print(f"Remaining goals: {env.goals}")

        step += 1

    # Compute optimal steps (hypothetical, assuming original assignment)
    total_opt = 0
    for start in init_positions:
        dists = [shortest_path_length(start, g, env) for g in goal_positions]
        total_opt += min(dists) if dists else 0

    failed = step >= max_steps
    print(f"\nRun finished in {step} steps. Collisions: {collisions}. Failed: {failed}")
    return step, total_opt, failed, collisions

if __name__ == "__main__":
    steps, optimal, failed, collisions = run()
    print(f"\n✅ Task completed!")
    print(f"Optimal: {optimal}, Steps: {steps}, Failed: {failed}, Collisions: {collisions}")
