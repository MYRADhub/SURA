import argparse
import csv
from core.environment import GridWorld
from core.utils import shortest_path_length, select_direction_opt

def compute_greedy_rankings(env):
    """For each agent, sort goals by ascending distance."""
    rankings = []
    for agent_pos in env.agents:
        dists = []
        for idx, goal_pos in enumerate(env.goals):
            dist = shortest_path_length(agent_pos, goal_pos, env)
            dists.append((dist, idx))
        dists.sort()
        rankings.append([chr(65 + idx) for _, idx in dists])
    return rankings

def resolve_conflicts(rankings, active):
    """Each agent gets its closest available goal; conflicts resolved by lower index."""
    assigned = [r[0] if r else None for r in rankings]
    positions = [0] * len(rankings)

    while True:
        goal_to_agents = {}
        for idx, g in enumerate(assigned):
            if active[idx] and g:
                goal_to_agents.setdefault(g, []).append(idx)

        conflicts = [agents for agents in goal_to_agents.values() if len(agents) > 1]
        if not conflicts:
            break

        for group in conflicts:
            group.sort()
            for loser in group[1:]:
                positions[loser] += 1
                if positions[loser] < len(rankings[loser]):
                    assigned[loser] = rankings[loser][positions[loser]]
                else:
                    assigned[loser] = None
    return assigned

def run(config_path, log_path="data/greedy_log.csv", max_steps=50):
    env = GridWorld(config_path)
    num_agents = len(env.agents)
    active = [True] * num_agents
    step = 0
    collisions = 0
    log_rows = []
    agent_ids = list(range(1, num_agents + 1))

    total_opt = sum(
        min([shortest_path_length(start, g, env) for g in env.goals if g], default=0)
        for start in env.agents if start
    )

    print(f"Initial agent positions: {env.agents}")
    print(f"Goals: {env.goals}")
    print(f"Obstacles: {env.obstacles}")

    agent_positions = env.agents[:]
    target_goals = [None] * num_agents

    while any(active) and step < max_steps:
        print(f"\n--- Step {step} ---")
        proposals = agent_positions[:]

        # Phase 1: Ranking
        rankings = compute_greedy_rankings(env)
        proposed_goals = resolve_conflicts(rankings, active)

        # Phase 2: Move Selection
        for i in range(num_agents):
            if not active[i] or not proposed_goals[i]:
                continue
            goal_char = proposed_goals[i]
            direction = select_direction_opt(agent_positions[i], goal_char, env.goals, env)
            if direction:
                new_pos = env.move_agent(agent_positions[i], direction)
                log_rows.append({
                    "step": step,
                    "agent_id": agent_ids[i],
                    "position_before": agent_positions[i],
                    "position_after": new_pos,
                    "chosen_direction": direction,
                    "target_goal": goal_char
                })
                proposals[i] = new_pos
            else:
                proposals[i] = agent_positions[i]

        # Phase 3: Collision resolution
        new_positions = proposals[:]
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if new_positions[i] == new_positions[j] and active[i] and active[j]:
                    collisions += 1
                    new_positions[j] = agent_positions[j]

        agent_positions = new_positions
        env.agents = new_positions[:]
        target_goals = proposed_goals[:]

        # Phase 4: Goal claiming
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

        step += 1

    failed = step >= max_steps
    print(f"\n✅ Finished in {step} steps. Collisions: {collisions}. Failed: {failed}")

    if log_rows:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)

    return step, total_opt, failed, collisions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run centralized greedy agent.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    steps, optimal, failed, collisions = run(config_path=args.config)
    print(f"\n📊 Greedy Results:\nOptimal: {optimal}\nSteps: {steps}\nFailed: {failed}\nCollisions: {collisions}")
