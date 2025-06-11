import time
import json
import re
from core.environment import GridWorld
from core.prompt import build_yesno_prompt_unassigned_com_unstructured
from core.request import send_image_to_model_openai_logprobs
from core.plot import plot_grid_unassigned
from core.utils import shortest_path_length

def extract_yes_logprob(logprobs):
    """
    Find the logprob of the "YES" token from the model output.
    Returns -inf if "YES" is not among the top_logprobs at any step.
    """
    if not logprobs:
        return float("-inf")
    
    print("Extracting logprob for 'YES' from logprobs...")
    # print(logprobs)

    for token_info in logprobs:
        token_str = token_info.token.strip().upper()
        if token_str in {"YES", "NO"}:
            for top in token_info.top_logprobs:
                if top.token.strip().upper() == "YES":
                    return top.logprob
    return float("-inf")


def parse_json_response(text):
    """Safely parse a JSON blob from response text."""
    try:
        json_start = text.index('{')
        json_str = text[json_start:]
        parsed = json.loads(json_str)
        move = parsed.get("move", "").strip().upper()
        target = parsed.get("target", "").strip().upper()
        explanation = parsed.get("explanation", "").strip()
        return move, target, explanation
    except Exception as e:
        print(f"❌ Failed to parse JSON: {e}")
        with open("fails.txt", "a") as f:
            f.write(f"Failed to parse JSON: {e}\n")
            f.write(f"Text: {text}\n\n")
        return None, None, ""

def run(
    obstacles={(2, 2), (3, 3), (4, 1)},
    grid_size=6,
    image_path="data/grid.png",
    log_path="data/agent_step_logs.csv",
    max_steps=30,
    num_agents=3,
    agent_starts: list[tuple[int, int]] = None,
    goal_positions: list[tuple[int, int]] = None
):
    env = GridWorld(grid_size, obstacles=obstacles)
    if agent_starts and goal_positions:
        env.initialize_agents_goals_custom(agents=agent_starts, goals=goal_positions)
    else:
        env.initialize_agents_goals(num_agents=num_agents)

    agent_positions = env.agents[:]
    agent_ids = list(range(1, num_agents + 1))
    active = [True] * num_agents
    visits = [{} for _ in range(num_agents)]
    memories = [[] for _ in range(num_agents)]
    target_goals = [None for _ in range(num_agents)]
    step = 0
    collisions = 0

    total_opt = sum(
        min([shortest_path_length(start, g, env) for g in env.goals if g is not None], default=0)
        for start in env.agents if start is not None
    )

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
                prompt = build_yesno_prompt_unassigned_com_unstructured(
                    agent_id=agent_ids[i],
                    agent_pos=agent_positions[i],
                    goal_positions=env.goals,
                    other_agents=other_infos,
                    grid_size=grid_size,
                    obstacles=obstacles,
                    direction=d,
                    memory=memories[i],
                    visits=visits[i],
                    agent_targets=target_goals
                )
                time.sleep(0.5)
                response_text, logprobs = send_image_to_model_openai_logprobs(image_path, prompt, temperature=0.0000001)
                # print(f"Agent {agent_ids[i]} response: {response_text}")
                score = extract_yes_logprob(logprobs)
                if score == float('-inf'):
                    print(f"Agent {agent_ids[i]} received no valid logprob for direction {d}")
                else:
                    print(f"Agent {agent_ids[i]} logprob for direction {d}: {score}")
                scores[d] = score

                move, target, explanation = parse_json_response(response_text)
                if target:
                    target_goals[i] = target
                    print(f"Agent {agent_ids[i]} declares target: {target}")
                if explanation:
                    print(f"Explanation: {explanation}")

            if scores:
                best = max(scores, key=scores.get)
                print(f"Agent {agent_ids[i]} chooses direction {best} with logprob {scores[best]}")
                print(f"Agent {agent_ids[i]} current target goal: {target_goals[i]}")
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
                    new_positions[j] = agent_positions[j]

        agent_positions = new_positions
        env.agents = agent_positions[:]
        print(f"Agent positions after moves: {agent_positions}")

        # Goal claiming
        to_remove = []
        claimed_goals = []
        for i in range(num_agents):
            if active[i] and agent_positions[i] in env.goals:
                print(f"Agent {agent_ids[i]} reached a goal at {agent_positions[i]}")
                active[i] = False
                to_remove.append(i)
                claimed_goals.append(agent_positions[i])

        for i in to_remove:
            agent_positions[i] = None
            env.agents[i] = None
            target_goals[i] = None

        for goal in claimed_goals:
            if goal in env.goals:
                env.goals[env.goals.index(goal)] = None

        print(f"Active agents: {[agent_ids[i] for i in range(num_agents) if active[i] and agent_positions[i] is not None]}")
        print(f"Remaining goals: {env.goals}")

        step += 1

    failed = step >= max_steps
    print(f"\nRun finished in {step} steps. Collisions: {collisions}. Failed: {failed}")
    return step, total_opt, failed, collisions

if __name__ == "__main__":
    steps, optimal, failed, collisions = run(
        grid_size=8,
        num_agents=2,
        agent_starts=[(7, 0), (4, 6)],
        goal_positions=[(2, 6), (0, 0)]
    )
    print(f"\n✅ Task completed!")
    print(f"Optimal: {optimal}, Steps: {steps}, Failed: {failed}, Collisions: {collisions}")
