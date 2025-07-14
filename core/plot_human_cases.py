import os
import numpy as np
import matplotlib.pyplot as plt
from core.environment import GridWorld
from collections import deque
from PIL import Image

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

def compute_distance_table(env):
    """
    Returns a dict: {agent_idx: {goal_idx: distance}}
    """
    distances = {}
    for i, agent in enumerate(env.agents):
        if agent is None:
            continue
        distances[i] = {}
        for j, goal in enumerate(env.goals):
            if goal is None:
                continue
            dist = shortest_path_length(agent, goal, env)
            distances[i][j] = dist
    return distances

def plot_grid_for_humans(env: GridWorld, image_path="data/grid_human.png"):
    grid = np.ones((env.size, env.size, 3))
    for pos in env.obstacles:
        grid[pos] = [0.0, 0.0, 0.0]

    goal_color = [1.0, 0.0, 0.0]
    for goal in env.goals:
        if goal is not None:
            grid[goal] = goal_color

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, extent=[0, env.size, 0, env.size], origin='lower')

    # Agents
    agent_color = [0.0, 0.4, 1.0]
    for idx, agent in enumerate(env.agents):
        if agent is not None:
            r, c = agent
            circle = plt.Circle((c + 0.5, r + 0.5), 0.45, color=agent_color, zorder=3)
            ax.add_patch(circle)

    # Grid lines
    for x in range(env.size + 1):
        ax.axhline(x, color='gray', linewidth=0.5)
        ax.axvline(x, color='gray', linewidth=0.5)

    ax.set_xticks(np.arange(env.size) + 0.5)
    ax.set_yticks(np.arange(env.size) + 0.5)
    ax.set_xticklabels([f"{i}" for i in range(env.size)])
    ax.set_yticklabels([f"{i}" for i in range(env.size)])
    ax.tick_params(axis='both', which='both', length=0)

    # Cell labels
    total_cells = env.size * env.size
    num_digits = len(str(total_cells))
    cell_counter = 1

    for r in range(env.size):
        for c in range(env.size):
            pos = (r, c)
            label = ""
            if pos in env.obstacles:
                label = "O"
            elif any(pos == a for a in env.agents if a is not None):
                idx = next(i for i, a in enumerate(env.agents) if a == pos)
                label = f"{idx + 1}"
            elif any(pos == g for g in env.goals if g is not None):
                idx = next(i for i, g in enumerate(env.goals) if g == pos)
                label = chr(65 + idx)

            if label:       # Agent / goal / obstacle
                ax.text(c + 0.5, r + 0.5, label,
                        color="white", fontsize=12, ha='center', va='center', weight='bold')
            else:           # Empty cell → show padded number
                padded_num = str(cell_counter).zfill(num_digits)
                ax.text(c + 0.5, r + 0.5, padded_num,
                        color="gray", fontsize=8, ha='center', va='center', alpha=0.6)
            cell_counter += 1

    # Distance table
    distances = compute_distance_table(env)
    agents = list(distances.keys())
    goals = sorted({g for d in distances.values() for g in d})
    table_data = []
    for a in agents:
        row = [distances[a].get(g, '∞') for g in goals]
        table_data.append(row)

    goal_labels = [chr(65 + g) for g in goals]
    agent_labels = [f"Agent {i+1}" for i in agents]

    fig_table, ax_table = plt.subplots(figsize=(len(goal_labels) + 1, len(agent_labels)))
    ax_table.axis('off')
    tbl = ax_table.table(cellText=table_data, colLabels=goal_labels,
                         rowLabels=agent_labels, loc='center', cellLoc='center')
    tbl.scale(1, 1.5)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    # Save both visuals side-by-side
    base = os.path.splitext(image_path)[0]
    grid_img = base + "_grid.png"
    table_img = base + "_table.png"
    plt.savefig(table_img, bbox_inches='tight')
    fig.savefig(grid_img, bbox_inches='tight')
    plt.close(fig)
    plt.close(fig_table)

    print(f"Saved grid to {grid_img}")
    print(f"Saved table to {table_img}")


    # Merge both images side-by-side
    combined_img_path = base + "_combined.png"

    img1 = Image.open(grid_img)
    img2 = Image.open(table_img)

    # Resize table to match height of grid
    if img1.height != img2.height:
        ratio = img1.height / img2.height
        new_width = int(img2.width * ratio)
        img2 = img2.resize((new_width, img1.height), Image.LANCZOS)

    # Combine horizontally
    combined = Image.new('RGB', (img1.width + img2.width, img1.height), (255, 255, 255))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))
    combined.save(combined_img_path)

    print(f"Combined image saved to {combined_img_path}")

    # Clean up individual images
    os.remove(grid_img)
    os.remove(table_img)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate human-readable grid and distance table.")
    parser.add_argument("--config", required=True, help="Path to a single config YAML file")
    parser.add_argument("--out", default="data/grid_human.png", help="Base path to save image(s)")
    args = parser.parse_args()

    env = GridWorld(args.config)
    plot_grid_for_humans(env, image_path=args.out)
