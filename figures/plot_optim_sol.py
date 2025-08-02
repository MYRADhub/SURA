import argparse
import matplotlib.pyplot as plt
import numpy as np
from core.environment import GridWorld
from core.find_optim_sol import compute_distance_matrix, find_best_assignment

AGENT_COLORS = ['#007bff', '#e94f37', '#44af69', '#f5cb5c', '#9966cc', '#b83b5e', '#2f4858']

def find_path_bfs(start, goal, env):
    from collections import deque
    visited = set()
    queue = deque([(start, [start])])
    while queue:
        pos, path = queue.popleft()
        if pos == goal:
            return path
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            next_pos = (nr, nc)
            if (0 <= nr < env.size and 0 <= nc < env.size and
                next_pos not in visited and
                next_pos not in env.obstacles):
                visited.add(next_pos)
                queue.append((next_pos, path + [next_pos]))
    return [start]  # fallback

def plot_grid_with_assignment(env, assignment, output_path):
    grid = np.ones((env.size, env.size, 3))
    for pos in env.obstacles:
        grid[pos] = [0.0, 0.0, 0.0]
    goal_color = [1.0, 0.0, 0.0]
    for goal in env.goals:
        if goal is not None:
            grid[goal] = goal_color

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, extent=[0, env.size, 0, env.size], origin='lower')

    # Grid lines
    for x in range(env.size + 1):
        ax.axhline(x, color='gray', linewidth=0.5)
        ax.axvline(x, color='gray', linewidth=0.5)

    # Axis ticks
    ax.set_xticks(np.arange(env.size) + 0.5)
    ax.set_yticks(np.arange(env.size) + 0.5)
    ax.set_xticklabels([f"{i}" for i in range(env.size)])
    ax.set_yticklabels([f"{i}" for i in range(env.size)])
    ax.tick_params(axis='both', which='both', length=0)

    # Add border directions
    margin = 0.5
    ax.axvline(env.size + margin, color='blue', linewidth=16, zorder=0)    # Right
    ax.axvline(-margin, color='yellow', linewidth=16, zorder=0)            # Left
    ax.axhline(env.size + margin, color='green', linewidth=16, zorder=0)   # Top
    ax.axhline(-margin, color='orange', linewidth=16, zorder=0)            # Bottom

    # Diagonal blockers
    for r in range(env.size - 1):
        for c in range(env.size - 1):
            tl, tr = (r + 1, c), (r + 1, c + 1)
            bl, br = (r, c), (r, c + 1)
            if tl in env.obstacles and br in env.obstacles and tr not in env.obstacles and bl not in env.obstacles:
                ax.plot([c + 0.5, c + 1.5], [r + 1.5, r + 0.5],
                        color='black', linewidth=10, solid_capstyle='round')
            elif tr in env.obstacles and bl in env.obstacles and tl not in env.obstacles and br not in env.obstacles:
                ax.plot([c + 1.5, c + 0.5], [r + 1.5, r + 0.5],
                        color='black', linewidth=10, solid_capstyle='round')

    # Obstacles annotation
    for r in range(env.size):
        for c in range(env.size):
            label = ""
            if (r, c) in env.obstacles:
                label = "O"
            elif any((r, c) == pos for pos in env.goals if pos is not None):
                idx = next(i for i, pos in enumerate(env.goals) if pos == (r, c))
                label = chr(65 + idx)
            if label:
                ax.text(
                    c + 0.5, r + 0.5, label,
                    color="white", fontsize=14, ha='center', va='center', weight='bold', zorder=4
                )

    # Draw agents as circles
    for idx, agent in enumerate(env.agents):
        if agent is not None:
            color = AGENT_COLORS[idx % len(AGENT_COLORS)]
            circle = plt.Circle((agent[1] + 0.5, agent[0] + 0.5), 0.45, color=color, zorder=5)
            ax.add_patch(circle)
            ax.text(
                agent[1] + 0.5, agent[0] + 0.5, f"{idx + 1}",
                color="white", fontsize=14, ha='center', va='center', weight='bold', zorder=6
            )

    # Draw arrows for the optimal assignment
    for agent_idx, goal_idx in enumerate(assignment):
        start = env.agents[agent_idx]
        goal = env.goals[goal_idx]
        path = find_path_bfs(start, goal, env)
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
        # Draw the path as a dashed line (excluding last point)
        if len(path) >= 2:
            xs, ys = zip(*path)
            ax.plot(
                [c + 0.5 for c in ys], [r + 0.5 for r in xs],
                color=color, linewidth=2, linestyle='--', alpha=0.85, zorder=2,
            )
            # Draw a thick arrow from the penultimate to the goal
            if len(path) >= 2:
                ax.arrow(
                    ys[-2] + 0.5, xs[-2] + 0.5,
                    ys[-1] - ys[-2], xs[-1] - xs[-2],
                    head_width=0.28, head_length=0.28,
                    fc=color, ec=color, zorder=3, length_includes_head=True,
                    alpha=0.9, linewidth=4
                )

    ax.set_xlim([-margin, env.size + margin])
    ax.set_ylim([-margin, env.size + margin])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved optimal assignment plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot grid and overlay optimal assignment solution")
    parser.add_argument('--config', required=True, help='YAML config file (GridWorld)')
    parser.add_argument('--output', default='optim_sol.png', help='Output image path')
    args = parser.parse_args()

    env = GridWorld(args.config)
    distances = compute_distance_matrix(env)
    assignment, cost = find_best_assignment(distances)

    print("Assignment:")
    for agent_idx, goal_idx in enumerate(assignment):
        print(f"  Agent {agent_idx + 1} → Goal {chr(65 + goal_idx)} (dist={distances[agent_idx][goal_idx]})")
    print(f"Optimal cost: {cost}")

    plot_grid_with_assignment(env, assignment, args.output)

if __name__ == "__main__":
    main()
