from collections import deque

def shortest_path_length(start, goal, env):
    if start == goal:
        return 0
    visited = set()
    queue = deque([(start, 0)])
    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == goal:
            return dist
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            next_pos = (nr, nc)
            if (
                0 <= nr < env.size and 0 <= nc < env.size and
                next_pos not in visited and
                next_pos not in env.obstacles
            ):
                visited.add(next_pos)
                queue.append((next_pos, dist + 1))
    return float('inf')  # No path found

from collections import deque

def is_reachable(grid_size, start, goal, obstacles):
    if start == goal:
        return True
    visited = set()
    queue = deque([start])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                neighbor = (nr, nc)
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    queue.append(neighbor)
    return False

def select_direction_opt(agent_pos, declared_goal, goal_positions, env):
    """
    Select the direction that reduces the distance to the target goal the fastest.
    """
    if not declared_goal:
        return None

    goal_index = ord(declared_goal.upper()) - 65
    if goal_index >= len(goal_positions) or goal_positions[goal_index] is None:
        return None

    target_goal = goal_positions[goal_index]
    best_dir = None
    best_dist = float('inf')

    directions = {
        "up": (1, 0),
        "down": (-1, 0),
        "left": (0, -1),
        "right": (0, 1)
    }

    for dir_str, (dr, dc) in directions.items():
        new_pos = (agent_pos[0] + dr, agent_pos[1] + dc)
        if env.is_valid(new_pos):
            dist = shortest_path_length(new_pos, target_goal, env)
            if dist < best_dist:
                best_dist = dist
                best_dir = dir_str
    return best_dir