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

