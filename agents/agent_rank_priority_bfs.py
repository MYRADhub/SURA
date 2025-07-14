import time
import json
import csv
from core.environment import GridWorld
from core.prompt import (
    build_target_ranking_prompt,
)
from core.request import send_image_to_model_openai_logprobs
from core.plot import plot_grid_unassigned_labeled
from core.utils import shortest_path_length, select_direction_opt
import re
import argparse
import random

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
    time.sleep(0.5)
    response, _ = send_image_to_model_openai_logprobs(image_path, prompt, model="gpt-4.1", temperature=0.0000001)

    raw_ranking, explanation, reasoning = parse_ranking_response(response)

    # Filter invalid goals
    valid_goals = set(chr(65 + i) for i in range(len(goal_positions)) if goal_positions[i] is not None)
    seen = set()
    filtered_ranking = []

    for g in raw_ranking:
        if g in valid_goals and g not in seen:
            filtered_ranking.append(g)
            seen.add(g)

    # Fallback: choose random goal if nothing valid
    if not filtered_ranking and valid_goals:
        fallback = random.choice(list(valid_goals))
        filtered_ranking = [fallback]
        print(f"⚠️ Agent {agent_id} had no valid ranking. Random fallback: {fallback}")

    print(f"Agent {agent_id} ranking: {filtered_ranking}")
    if reasoning:
        print(f"Reasoning: {reasoning}")
    if explanation:
        print(f"Summary: {explanation}")

    if filtered_ranking:
        top = filtered_ranking[0]
        target_memory.append((step, top, explanation))
        if len(target_memory) > 5:
            target_memory[:] = target_memory[-5:]

    return filtered_ranking, explanation, reasoning

def resolve_conflicts(agent_rankings, active_agents):
    final_goals = [rank[0] if rank else None for rank in agent_rankings]
    positions = [0 for _ in agent_rankings]

    while True:
        goal_to_agents = {}
        for idx, goal in enumerate(final_goals):
            if active_agents[idx] and goal:
                goal_to_agents.setdefault(goal, []).append(idx)

        conflicts_exist = any(len(lst) > 1 for lst in goal_to_agents.values())
        if not conflicts_exist:
            break

        for goal, agents in goal_to_agents.items():
            if len(agents) <= 1:
                continue
            agents.sort()  # Ensure lower index agent wins
            for loser_idx in agents[1:]:
                positions[loser_idx] += 1
                if positions[loser_idx] < len(agent_rankings[loser_idx]):
                    final_goals[loser_idx] = agent_rankings[loser_idx][positions[loser_idx]]
                else:
                    final_goals[loser_idx] = None
    return final_goals

def run(
    image_path="data/grid.png",
    log_path="data/agent_rank_logs.csv",
    max_steps=100,
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

        # Resolve conflicts deterministically
        proposed_goals = resolve_conflicts(agent_rankings, active)

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

            best = select_direction_opt(
                agent_pos,
                new_target,
                env.goals,
                env
            )
            explanation = "Chose shortest path step deterministically."

            if best:
                proposals[i] = env.move_agent(agent_pos, best)
                log_rows.append({
                    "step": step,
                    "agent_id": agent_id,
                    "position_before": agent_pos,
                    "position_after": proposals[i],
                    "chosen_direction": best,
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
