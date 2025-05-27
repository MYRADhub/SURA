import matplotlib.pyplot as plt
import numpy as np
from core.environment import GridWorld

def plot_grid(env: GridWorld, image_path="data/grid.png"):
    grid = np.ones((env.size, env.size, 3))

    # Draw obstacles (black)
    for pos in env.obstacles:
        grid[pos] = [0.0, 0.0, 0.0]  # Black

    # Draw agents (blue)
    agent_color = [0.0, 0.4, 1.0]  # Blue
    for agent in env.agents:
        grid[agent] = agent_color

    # Draw goals (red)
    goal_color = [1.0, 0.0, 0.0]  # Red
    for goal in env.goals:
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
    ax.set_xticklabels([f"col {i}" for i in range(env.size)])
    ax.set_yticklabels([f"row {i}" for i in range(env.size)])
    ax.tick_params(axis='both', which='both', length=0)

    # Add text annotations
    for r in range(env.size):
        for c in range(env.size):
            label = ""
            if (r, c) in env.obstacles:
                label = "OBS"
            elif (r, c) in env.agents:
                idx = env.agents.index((r, c))
                label = f"A{idx + 1}"
            elif (r, c) in env.goals:
                idx = env.goals.index((r, c))
                label = f"G{idx + 1}"
            if label:
                ax.text(
                    c + 0.5, r + 0.5, label,
                    color="white",
                    fontsize=12,
                    ha='center',
                    va='center',
                    weight='bold'
                )

    # Border directions
    margin = 0.5
    ax.axvline(env.size + margin, color='blue', linewidth=16)    # Right
    ax.axvline(0 - margin, color='yellow', linewidth=16)         # Left
    ax.axhline(env.size + margin, color='green', linewidth=16)   # Top
    ax.axhline(0 - margin, color='orange', linewidth=16)         # Bottom

    ax.set_xlim([-margin, env.size + margin])
    ax.set_ylim([-margin, env.size + margin])
    plt.axis('off')
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()


# Test usage
if __name__ == "__main__":
    env = GridWorld(size=8, obstacles={(2, 2), (3, 3), (5, 5)})
    env.initialize_agents_goals(num_agents=4)
    plot_grid(env)
    print("Grid with arbitrary agents and obstacles saved.")
