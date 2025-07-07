import itertools
from core.environment import GridWorld
from core.utils import shortest_path_length

def compute_distance_matrix(env):
    """
    Returns a matrix: distances[i][j] = steps from agent i to goal j
    """
    distances = []
    for agent_pos in env.agents:
        row = []
        for goal_pos in env.goals:
            d = shortest_path_length(agent_pos, goal_pos, env)
            row.append(d)
        distances.append(row)
    return distances

def find_best_assignment(distances):
    """
    distances: list of list of distances[agent][goal]
    Returns: (assignment, cost)
      assignment: list of goal indices assigned to each agent (in order)
      cost: the max distance
    """
    num_agents = len(distances)
    all_goal_indices = list(range(num_agents))
    best_cost = float('inf')
    best_assignment = None

    for perm in itertools.permutations(all_goal_indices):
        # perm is a tuple of goal indices assigned to agents in order
        agent_dists = [distances[i][goal_idx] for i, goal_idx in enumerate(perm)]
        cost = max(agent_dists)
        if cost < best_cost:
            best_cost = cost
            best_assignment = perm

    return best_assignment, best_cost

def main(config_path):
    env = GridWorld(config_path)
    distances = compute_distance_matrix(env)
    assignment, cost = find_best_assignment(distances)

    print(f"Optimal assignment (Agent index -> Goal index):")
    for agent_idx, goal_idx in enumerate(assignment):
        goal_letter = chr(ord('A') + goal_idx)
        print(f"  Agent {agent_idx+1}: Goal {goal_letter} (distance {distances[agent_idx][goal_idx]})")
    print(f"\nOptimal cost (max distance): {cost}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
