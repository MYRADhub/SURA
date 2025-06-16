import time
import json
import csv
from core.environment import GridWorld
from core.prompt import (
    build_target_selection_prompt,
    build_direction_selection_prompt,
)
from core.request import send_image_to_model_openai_logprobs
from core.plot import plot_grid_unassigned_labeled
from core.utils import shortest_path_length
from core.request import send_image_to_model_openai_logprobs
import re

def extract_yes_logprob(logprobs):
    if not logprobs:
        return float("-inf")
    for token_info in logprobs:
        token_str = token_info.token.strip()
        if token_str in {"YES", "NO"}:
            for top in token_info.top_logprobs:
                if top.token.strip() == "YES":
                    return top.logprob
    return float("-inf")

def extract_top_goals(logprobs):
    goal_scores = {}
    for entry in logprobs:
        for tok in entry.top_logprobs:
            t = tok.token.strip()
            if t in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                goal_scores[t] = max(goal_scores.get(t, float('-inf')), tok.logprob)
    top2 = sorted(goal_scores.items(), key=lambda x: -x[1])[:2]
    return top2

def parse_target_response(text):
    try:
        # Extract JSON block between triple backticks
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Fallback: try to find first '{'
            json_start = text.index('{')
            json_str = text[json_start:]
        parsed = json.loads(json_str)
        target = parsed.get("target", "").strip().upper()
        explanation = parsed.get("explanation", "").strip()
        return target, explanation
    except Exception as e:
        print(f"Failed to parse target JSON: {e}")
        with open("fails.txt", "a") as f:
            f.write(f"[Target Parsing Error] {e}\n{text}\n\n")
        return None, ""

def parse_move_response(text):
    try:
        # Extract JSON block between triple backticks
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Fallback: try to find first '{'
            json_start = text.index('{')
            json_str = text[json_start:]
        parsed = json.loads(json_str)
        move = parsed.get("move", "").strip().upper()
        explanation = parsed.get("explanation", "").strip()
        return move, explanation
    except Exception as e:
        print(f"Failed to parse move JSON: {e}")
        with open("fails.txt", "a") as f:
            f.write(f"[Move Parsing Error] {e}\n{text}\n\n")
        return None, ""

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

    agent_positions = env.agents[:]
    agent_ids = list(range(1, num_agents + 1))
    active = [True] * num_agents
    visits = [{} for _ in range(num_agents)]
    memories = [[] for _ in range(num_agents)]
    target_goals = [None for _ in range(num_agents)]
    step = 0
    collisions = 0
    log_rows = []

    total_opt = sum(
        min([shortest_path_length(start, g, env) for g in env.goals if g is not None], default=0)
        for start in env.agents if start is not None
    )

    print(f"Initial agent positions: {agent_positions}")
    print(f"Goal positions: {env.goals}")
    print(f"Obstacles: {obstacles}")

    while any(active) and step < max_steps:
        print(f"\n--- Step {step} ---")
        plot_grid_unassigned_labeled(env, image_path=image_path)
        proposals = agent_positions[:]
        proposed_goals = target_goals[:]

        for i in range(num_agents):
            if not active[i] or agent_positions[i] is None:
                continue

            agent_id = agent_ids[i]
            agent_pos = agent_positions[i]
            visits[i][agent_pos] = visits[i].get(agent_pos, 0) + 1
            other_infos = [
                (agent_ids[j], agent_positions[j])
                for j in range(num_agents)
                if j != i and agent_positions[j] is not None
            ]

            # Target selection phase
            target_prompt = build_target_selection_prompt(
                agent_id=agent_id,
                agent_pos=agent_pos,
                goal_positions=env.goals,
                other_agents=other_infos,
                grid_size=grid_size,
                obstacles=obstacles,
                memory=memories[i],
                visits=visits[i],
                agent_targets=target_goals
            )
            time.sleep(0.5)
            target_response, _ = send_image_to_model_openai_logprobs(image_path, target_prompt, temperature=0.0000001)
            print(f"Agent {agent_id} target selection response:\n{target_response}")
            new_target, target_explanation = parse_target_response(target_response)
            if new_target:
                proposed_goals[i] = new_target
                print(f"Agent {agent_id} selected target: {new_target}")
            if target_explanation:
                print(f"Target reasoning: {target_explanation}")

            # Move selection phase
            valid = env.get_valid_actions(agent_pos)
            scores = {}
            logprobs_by_dir = {}
            move_explanations = {}

            for direction in valid:
                dir_prompt = build_direction_selection_prompt(
                    agent_id=agent_id,
                    agent_pos=agent_pos,
                    declared_goal=target_goals[i],
                    goal_positions=env.goals,
                    other_agents=other_infos,
                    grid_size=grid_size,
                    obstacles=obstacles,
                    direction=direction,
                    memory=memories[i],
                    visits=visits[i],
                    agent_targets=target_goals
                )
                time.sleep(0.5)
                response_text, logprobs = send_image_to_model_openai_logprobs(image_path, dir_prompt, temperature=0.0000001)
                score = extract_yes_logprob(logprobs)
                scores[direction] = score
                logprobs_by_dir[direction] = logprobs
                move, explanation = parse_move_response(response_text)
                move_explanations[direction] = (move, explanation)

            if scores:
                best = max(scores, key=scores.get)
                move, explanation = move_explanations[best]
                print(f"Agent {agent_id} moves {best} toward goal {proposed_goals[i]}")
                print(f"Explanation: {explanation}")
                proposals[i] = env.move_agent(agent_pos, best)
                top_goals = extract_top_goals(logprobs_by_dir[best])

                log_rows.append({
                    "step": step,
                    "agent_id": agent_id,
                    "position_before": agent_pos,
                    "position_after": proposals[i],
                    "chosen_direction": best,
                    "logprob_yes": f"{scores[best]:.5f}",
                    "target_goal": target_goals[i],
                    "goal_top1": top_goals[0][0] if len(top_goals) > 0 else "",
                    "goal_top1_logprob": f"{top_goals[0][1]:.5f}" if len(top_goals) > 0 else "",
                    "goal_top2": top_goals[1][0] if len(top_goals) > 1 else "",
                    "goal_top2_logprob": f"{top_goals[1][1]:.5f}" if len(top_goals) > 1 else "",
                    "explanation": explanation,
                })

                memories[i].append((agent_pos[0], agent_pos[1], best, proposals[i][0], proposals[i][1]))
                if len(memories[i]) > 5:
                    memories[i] = memories[i][-5:]
            else:
                print(f"Agent {agent_id} has no valid moves and stays at {agent_pos}")
                proposals[i] = agent_pos

        target_goals = proposed_goals[:]

        # Collision resolution
        new_positions = proposals[:]
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if new_positions[i] == new_positions[j] and agent_positions[i] is not None and agent_positions[j] is not None:
                    collisions += 1
                    print(f"Collision: Agent {agent_ids[i]} and Agent {agent_ids[j]} at {new_positions[i]}")
                    new_positions[j] = agent_positions[j]

        agent_positions = new_positions
        env.agents = agent_positions[:]

        # Goal claiming
        to_remove = []
        claimed_goals = []
        for i in range(num_agents):
            if active[i] and agent_positions[i] in env.goals:
                print(f"Agent {agent_ids[i]} reached goal at {agent_positions[i]}")
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

        print(f"Remaining agents: {[agent_ids[i] for i in range(num_agents) if active[i]]}")
        print(f"Remaining goals: {env.goals}")
        step += 1

    failed = step >= max_steps
    print(f"\nRun completed in {step} steps. Collisions: {collisions}. Failed: {failed}")
    if log_rows:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
    return step, total_opt, failed, collisions


if __name__ == "__main__":
    steps, optimal, failed, collisions = run(config_path="configs/case_9_2_greedy_agents.yaml",)
    print(f"\n✅ Done!\nOptimal: {optimal}, Steps: {steps}, Failed: {failed}, Collisions: {collisions}")
