from core.environment import GridWorld
from core.prompt import build_yesno_prompt_unassigned_com
from core.request import send_image_to_model_openai_logprobs_formatted
from core.plot import plot_grid_unassigned
from core.utils import shortest_path_length
from core.schema import OpenAIResponse
import csv
import re
import time

def extract_yes_logprob(logprobs):
    if not logprobs:
        return float('-inf')
    for item in logprobs[4].top_logprobs:
        if item.token.strip() == "YES":
            return item.logprob
    return float('-inf')

def extract_top_goals(logprobs):
    """
    Extract the top 2 goal tokens (A, B, C, ...) from logprobs.
    Returns list of (token, logprob), sorted.
    """
    goal_scores = {}
    for entry in logprobs:
        for tok in entry.top_logprobs:
            t = tok.token.strip()
            if t in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                goal_scores[t] = max(goal_scores.get(t, float('-inf')), tok.logprob)
    top2 = sorted(goal_scores.items(), key=lambda x: -x[1])[:2]
    return top2

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
    target_goals = [None for _ in range(num_agents)]  # NEW: tracks declared goals
    step = 0
    collisions = 0
    log_rows = []

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
            scores = {}  # direction -> score
            dir_logprobs = {}  # direction -> logprobs
            dir_targets = {}  # direction -> (target_goal, explanation)

            for d in valid:
                other_infos = [
                    (agent_ids[j], agent_positions[j])
                    for j in range(num_agents)
                    if j != i and agent_positions[j] is not None
                ]
                prompt = build_yesno_prompt_unassigned_com(
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
                response_text, logprobs = send_image_to_model_openai_logprobs_formatted(image_path, prompt, temperature=0.0000001)
                score = extract_yes_logprob(logprobs)
                scores[d] = score
                dir_logprobs[d] = logprobs

                target_goal = None
                explanation = ""
                try:
                    parsed = OpenAIResponse.model_validate_json(response_text)
                    target_goal = parsed.target_goal
                    explanation = parsed.explanation
                except Exception as e:
                    print(f"Warning: Failed to parse structured response: {e}")
                    try:
                        goal_match = re.search(r'"?target"?\s*[:=]\s*"?(?P<goal>[A-Z])"?', response_text, re.IGNORECASE)
                        if goal_match:
                            target_goal = goal_match.group("goal").upper()
                        exp_match = re.search(r'"?explanation"?\s*[:=]\s*"(.*?)"', response_text, re.IGNORECASE)
                        if exp_match:
                            explanation = exp_match.group(1)
                    except:
                        pass

                dir_targets[d] = (target_goal, explanation)
                if score > -5:
                    target_goals[i] = target_goal

            if scores:
                best = max(scores, key=scores.get)
                print(f"Agent {agent_ids[i]} chooses direction {best} with logprob {scores[best]}")
                print(f"Agent {agent_ids[i]} current target goal: {target_goals[i]}")
                proposals[i] = env.move_agent(agent_positions[i], best)
                top_goals = extract_top_goals(dir_logprobs[best])
                goal_info = dir_targets[best]
                log_rows.append({
                    "step": step,
                    "agent_id": agent_ids[i],
                    "position_before": agent_positions[i],
                    "position_after": proposals[i],
                    "chosen_direction": best,
                    "logprob_yes": f"{scores[best]:.5f}",
                    "target_goal": goal_info[0],
                    "goal_top1": top_goals[0][0] if len(top_goals) > 0 else "",
                    "goal_top1_logprob": f"{top_goals[0][1]:.5f}" if len(top_goals) > 0 else "",
                    "goal_top2": top_goals[1][0] if len(top_goals) > 1 else "",
                    "goal_top2_logprob": f"{top_goals[1][1]:.5f}" if len(top_goals) > 1 else "",
                    "explanation": goal_info[1],
                })
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
        claimed_goals = []
        for i in range(num_agents):
            if not active[i] or agent_positions[i] is None:
                continue
            if agent_positions[i] in env.goals:
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
    if log_rows:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
    return step, total_opt, failed, collisions

if __name__ == "__main__":
    steps, optimal, failed, collisions = run(grid_size=8,
        num_agents=2,
        agent_starts=[(2, 0), (2, 3)],
        goal_positions=[(1, 4), (7, 7)],
        obstacles={(3, 3), (4, 4), (2, 5), (5, 2), (6, 6)})
    print(f"\n✅ Task completed!")
    print(f"Optimal: {optimal}, Steps: {steps}, Failed: {failed}, Collisions: {collisions}")
