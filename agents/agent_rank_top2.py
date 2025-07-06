import time
import json
import csv
from core.environment import GridWorld
from core.prompt import (
    build_target_ranking_prompt,
    build_direction_selection_prompt,
    build_negotiation_prompt
)
from core.request import send_image_to_model_openai_logprobs, send_text_to_model_openai
from core.plot import plot_grid_unassigned_labeled
from core.utils import shortest_path_length, select_direction_opt
from core.request import send_image_to_model_openai_logprobs
import re
import argparse

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

def parse_ranking_response(text):
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        json_str = match.group(1) if match else text[text.index("{"):]
        parsed = json.loads(json_str)

        reasoning = parsed.get("reasoning", "").strip()
        explanation = parsed.get("explanation", "").strip()
        ranking = [g.strip().upper() for g in parsed.get("ranking", []) if isinstance(g, str)]
        return ranking, explanation, reasoning
    except Exception as e:
        print(f"Failed to parse ranking JSON: {e}")
        with open("fails.txt", "a") as f:
            f.write(f"[Ranking Parsing Error] {e}\n{text}\n\n")
        return [], "", ""

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

def select_target(
    agent_id,
    agent_pos,
    goal_positions,
    other_agents,
    grid_size,
    obstacles,
    memory,
    visits,
    agent_targets,
    target_memory,
    image_path,
    step,
    distances
):
    prompt = build_target_ranking_prompt(
        agent_id=agent_id,
        agent_pos=agent_pos,
        goal_positions=goal_positions,
        other_agents=other_agents,
        grid_size=grid_size,
        obstacles=obstacles,
        memory=memory,
        visits=visits,
        agent_targets=agent_targets,
        target_memory=target_memory,
        distances=distances
    )
    # with open("debug_prompt.txt", "w") as f:
    #     f.write(prompt)
    # # stop the program here to inspect the prompt
    # input("Press Enter to continue...")
    time.sleep(0.5)
    response, _ = send_image_to_model_openai_logprobs(image_path, prompt, model="gpt-4.1", temperature=0.0000001)
    print(f"Agent {agent_id} ranking response:\n...{response[-200:]}")

    ranking, explanation, reasoning = parse_ranking_response(response)

    if ranking:
        print(f"Agent {agent_id} ranking: {ranking}")
    if reasoning:
        print(f"Reasoning: {reasoning}")
    if explanation:
        print(f"Summary: {explanation}")

    if ranking:
        top = ranking[0]
        target_memory.append((step, top, explanation))
        if len(target_memory) > 5:
            target_memory[:] = target_memory[-5:]

    return ranking, explanation, reasoning

def select_direction(
    agent_id,
    agent_pos,
    declared_goal,
    goal_positions,
    other_agents,
    grid_size,
    obstacles,
    memory,
    visits,
    agent_targets,
    image_path,
    env
):
    valid = env.get_valid_actions(agent_pos)
    scores = {}
    logprobs_by_dir = {}
    explanations = {}

    for direction in valid:
        prompt = build_direction_selection_prompt(
            agent_id=agent_id,
            agent_pos=agent_pos,
            declared_goal=declared_goal,
            goal_positions=goal_positions,
            other_agents=other_agents,
            grid_size=grid_size,
            obstacles=obstacles,
            direction=direction,
            memory=memory,
            visits=visits,
            agent_targets=agent_targets
        )
        time.sleep(0.5)
        response, logprobs = send_image_to_model_openai_logprobs(image_path, prompt, temperature=0.0000001)
        score = extract_yes_logprob(logprobs)
        scores[direction] = score
        logprobs_by_dir[direction] = logprobs
        move, explanation = parse_move_response(response)
        explanations[direction] = (move, explanation)

    if not scores:
        return None, None, None, {}

    best = max(scores, key=scores.get)
    move, explanation = explanations[best]
    print(f"Agent {agent_id} moves {best} toward goal {declared_goal}")
    # print(f"Explanation: {explanation}")

    return best, explanation, logprobs_by_dir[best], scores

def run_negotiation(env, conflict_tuple, agent_ids, agent_positions, goal_positions, distances, agent_rankings, max_rounds=4):
    print(f"\n--- Negotiation for conflict: {conflict_tuple} ---")
    i, j, goal = conflict_tuple
    id_i, id_j = agent_ids[i], agent_ids[j]
    proposal = None

    # Keep only top 2 goals per agent
    top_goals_i = agent_rankings[i][:2]
    top_goals_j = agent_rankings[j][:2]
    allowed_goals = sorted(set(top_goals_i + top_goals_j))
    allowed_indices = [ord(g) - 65 for g in allowed_goals]

    # Build reduced distances dict
    reduced_distances = {}
    for aid, idx in zip([id_i, id_j], [i, j]):
        reduced_distances[aid] = [
            (chr(65 + k), distances[aid][k]) for k in allowed_indices
        ]


    # Build reduced rankings dict
    reduced_rankings = {
        id_i: top_goals_i,
        id_j: top_goals_j
    }

    # For displaying what other agents are targeting
    agent_targets = {
        agent_ids[k]: agent_rankings[k][0] if agent_rankings[k] else None
        for k in range(len(agent_ids))
    }
    filtered_agent_targets = {
        aid: tgt
        for aid, tgt in agent_targets.items()
        if aid not in {id_i, id_j}
    }

    for round_number in range(1, max_rounds + 1):
        # Agent i's turn
        prompt_i = build_negotiation_prompt(
            self_id=id_i,
            self_pos=agent_positions[i],
            opponent_id=id_j,
            opponent_pos=agent_positions[j],
            goal_positions=goal_positions,
            distances=reduced_distances,
            rankings=reduced_rankings,
            agent_targets=filtered_agent_targets,
            conflicted_goal=goal,
            previous_proposal=proposal,
            round_number=round_number,
            max_rounds=max_rounds
        )
        print(f"Agent {id_i} negotiation prompt:\n{prompt_i[:200]}...")
        # save the prompt to a file for debugging
        # with open("debug_prompt.txt", "w") as f:
        #     f.write(prompt_i)
        # # stop the program here to inspect the prompt
        # input("Press Enter to continue...")
        response_i = send_text_to_model_openai(prompt_i, model="gpt-4.1", temperature=0.0000001)
        try:
            parsed_i = json.loads(re.search(r"\{.*\}", response_i, re.DOTALL).group())
        except:
            print("⚠️ Failed to parse Agent", id_i)
            return None
        if parsed_i["action"] == "accept":
            return parsed_i["proposal"]
        elif parsed_i["action"] == "reject":
            break
        proposal = parsed_i["proposal"]
        print(f"Agent {id_i} proposed: {proposal}")

        # Agent j's turn
        prompt_j = build_negotiation_prompt(
            self_id=id_j,
            self_pos=agent_positions[j],
            opponent_id=id_i,
            opponent_pos=agent_positions[i],
            goal_positions=goal_positions,
            distances=reduced_distances,
            rankings=reduced_rankings,
            agent_targets=filtered_agent_targets,
            conflicted_goal=goal,
            previous_proposal=proposal,
            round_number=round_number,
            max_rounds=max_rounds
        )
        print(f"Agent {id_j} negotiation prompt:\n{prompt_j[:200]}...")
        response_j = send_text_to_model_openai(prompt_j, model="gpt-4.1", temperature=0.0000001)
        try:
            parsed_j = json.loads(re.search(r"\{.*\}", response_j, re.DOTALL).group())
        except:
            print("⚠️ Failed to parse Agent", id_j)
            return None
        if parsed_j["action"] == "accept":
            return parsed_j["proposal"]
        elif parsed_j["action"] == "reject":
            break
        proposal = parsed_j["proposal"]
        print(f"Agent {id_j} proposed: {proposal}")
    if proposal:
        print(f"Final proposal accepted: {proposal}")
        return {f"Agent {id_i}": proposal[f"Agent {id_i}"], f"Agent {id_j}": proposal[f"Agent {id_j}"]}
    else:
        print("No agreement reached.")
    return None
  # fallback if no agreement

def run(
    image_path="data/grid.png",
    log_path="data/agent_rank_logs.csv",
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
    target_memories = [[] for _ in range(num_agents)]
    target_goals = [None for _ in range(num_agents)]
    step = 0
    collisions = 0
    log_rows = []
    agent_rankings = [[] for _ in range(num_agents)]

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
        proposed_goals = [None for _ in range(num_agents)]

        # ----------- Phase 1: Ranking ----------
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

            distances = {
                agent_ids[j]: [
                    shortest_path_length(agent_positions[j], goal, env) if agent_positions[j] and goal else float("inf")
                    for goal in env.goals
                ]
                for j in range(num_agents)
            }

            ranking, _, _ = select_target(
                agent_id=agent_id,
                agent_pos=agent_pos,
                goal_positions=env.goals,
                other_agents=other_infos,
                grid_size=grid_size,
                obstacles=obstacles,
                memory=memories[i],
                visits=visits[i],
                agent_targets=target_goals,
                target_memory=target_memories[i],
                image_path=image_path,
                step=step,
                distances=distances
            )
            agent_rankings[i] = ranking

        # ----------- Phase 2: Conflict Resolution ----------
        proposed_goals = [rank[0] if rank else None for rank in agent_rankings]

        goal_to_agents = {}
        for idx, tgt in enumerate(proposed_goals):
            if active[idx] and tgt:
                goal_to_agents.setdefault(tgt, []).append(idx)

        conflict_pairs = [
            (a1, a2, goal)
            for goal, agents in goal_to_agents.items()
            if len(agents) == 2
            for a1, a2 in [tuple(sorted(agents))]
        ]

        print("Conflict pairs detected:", conflict_pairs)

        for conflict in conflict_pairs:
            resolution = run_negotiation(
                env=env,
                conflict_tuple=conflict,
                agent_ids=agent_ids,
                agent_positions=agent_positions,
                goal_positions=env.goals,
                distances={
                    agent_ids[j]: [
                        shortest_path_length(agent_positions[j], goal, env) if agent_positions[j] and goal else float("inf")
                        for goal in env.goals
                    ]
                    for j in range(num_agents)
                },
                agent_rankings=agent_rankings
            )
            if resolution:
                for aid_str, tgt in resolution.items():
                    aid = int(aid_str.split()[-1])
                    idx = agent_ids.index(aid)
                    proposed_goals[idx] = tgt

        # ----------- Phase 3: Direction Selection & Movement ----------
        for i in range(num_agents):
            if not active[i] or agent_positions[i] is None:
                continue

            agent_id = agent_ids[i]
            agent_pos = agent_positions[i]
            new_target = proposed_goals[i]

            other_infos = [
                (agent_ids[j], agent_positions[j])
                for j in range(num_agents)
                if j != i and agent_positions[j] is not None
            ]

            best, explanation, logprobs, scores = select_direction(
                agent_id=agent_id,
                agent_pos=agent_pos,
                declared_goal=new_target,
                goal_positions=env.goals,
                other_agents=other_infos,
                grid_size=grid_size,
                obstacles=obstacles,
                memory=memories[i],
                visits=visits[i],
                agent_targets=target_goals,
                image_path=image_path,
                env=env
            )

            if best:
                proposals[i] = env.move_agent(agent_pos, best)
                top_goals = extract_top_goals(logprobs)
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
                proposals[i] = agent_pos

        target_goals = proposed_goals[:]

        # ----------- Collision Resolution ----------
        new_positions = proposals[:]
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if new_positions[i] == new_positions[j] and agent_positions[i] is not None and agent_positions[j] is not None:
                    collisions += 1
                    new_positions[j] = agent_positions[j]

        agent_positions = new_positions
        env.agents = agent_positions[:]

        # ----------- Goal Claiming ----------
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
    parser = argparse.ArgumentParser(description="Run agent rank simulation.")
    parser.add_argument("--config", type=str, default="configs/difficult/case_10_insane.yaml", help="Path to config YAML file")
    args = parser.parse_args()

    steps, optimal, failed, collisions = run(config_path=args.config)
    print(f"\n✅ Done!\nOptimal: {optimal}, Steps: {steps}, Failed: {failed}, Collisions: {collisions}")
