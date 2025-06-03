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
                label = "O"
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

        # Draw diagonal walls between diagonally adjacent obstacles
    for r in range(env.size - 1):
        for c in range(env.size - 1):
            # Check 2x2 square corners for diagonally placed obstacles
            tl = (r + 1, c)     # top-left
            tr = (r + 1, c + 1) # top-right
            bl = (r, c)         # bottom-left
            br = (r, c + 1)     # bottom-right

            if tl in env.obstacles and br in env.obstacles and tr not in env.obstacles and bl not in env.obstacles:
                # draw line from center of tl to br
                ax.plot(
                    [c + 0.5, c + 1.5],
                    [r + 1.5, r + 0.5],
                    color='black',
                    linewidth=10,
                    solid_capstyle='round'
                )
            elif tr in env.obstacles and bl in env.obstacles and tl not in env.obstacles and br not in env.obstacles:
                # draw line from center of tr to bl
                ax.plot(
                    [c + 1.5, c + 0.5],
                    [r + 1.5, r + 0.5],
                    color='black',
                    linewidth=10,
                    solid_capstyle='round'
                )

    plt.savefig(image_path, bbox_inches='tight')
    plt.close()

def plot_grid_unassigned(env: GridWorld, image_path="data/grid.png"):
    grid = np.ones((env.size, env.size, 3))

    # Draw obstacles (black)
    for pos in env.obstacles:
        grid[pos] = [0.0, 0.0, 0.0]

    # Draw agents (blue)
    agent_color = [0.0, 0.4, 1.0]
    for agent in env.agents:
        grid[agent] = agent_color

    # Draw goals (red)
    goal_color = [1.0, 0.0, 0.0]
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

    # Add text annotations: Agents as numbers, goals as letters
    for r in range(env.size):
        for c in range(env.size):
            label = ""
            if (r, c) in env.obstacles:
                label = "O"
            elif (r, c) in env.agents:
                idx = env.agents.index((r, c))
                label = f"{idx + 1}"  # Agent labels: 1, 2, 3...
            elif (r, c) in env.goals:
                idx = env.goals.index((r, c))
                label = chr(65 + idx)  # Goal labels: A, B, C...
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

    # Diagonal blockers between obstacles
    for r in range(env.size - 1):
        for c in range(env.size - 1):
            tl = (r + 1, c)
            tr = (r + 1, c + 1)
            bl = (r, c)
            br = (r, c + 1)
            if tl in env.obstacles and br in env.obstacles and tr not in env.obstacles and bl not in env.obstacles:
                ax.plot([c + 0.5, c + 1.5], [r + 1.5, r + 0.5], color='black', linewidth=10, solid_capstyle='round')
            elif tr in env.obstacles and bl in env.obstacles and tl not in env.obstacles and br not in env.obstacles:
                ax.plot([c + 1.5, c + 0.5], [r + 1.5, r + 0.5], color='black', linewidth=10, solid_capstyle='round')

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
    plot_grid_unassigned(env, image_path="data/grid_unassigned.png")
    print("Grid with arbitrary agents and obstacles saved.")
