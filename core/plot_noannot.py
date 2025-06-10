import matplotlib.pyplot as plt
import numpy as np
from core.environment import GridWorld

def plot_grid(env: GridWorld, image_path="grid.png"):
    grid = np.ones((env.size, env.size, 3))

    # Draw obstacles
    for pos in env.obstacles:
        grid[pos] = [0.4, 0.2, 0]  # Brown

    # Draw agents (grayscale)
    agent_colors = [
        [0.0, 0.0, 0.0],       # Black
        [0.5, 0.5, 0.5],       # Gray
        [0.2, 0.2, 0.2],       # Dark gray
        [0.7, 0.7, 0.7],       # Light gray
    ]

    for i, agent in enumerate(env.agents):
        color = agent_colors[i % len(agent_colors)]
        grid[agent] = color

    # Draw goals (warm colors)
    goal_colors = [
        [1.0, 0.0, 0.0],       # Red
        [1.0, 0.5, 0.0],       # Orange
        [1.0, 0.7, 0.3],       # Lighter orange
        [0.9, 0.3, 0.3],       # Salmon
    ]

    for i, goal in enumerate(env.goals):
        color = goal_colors[i % len(goal_colors)]
        grid[goal] = color

    # Draw grid
    plt.imshow(grid, extent=[0, env.size, 0, env.size], origin='lower')
    for x in range(env.size + 1):
        plt.axhline(x, color='gray', linewidth=0.5)
        plt.axvline(x, color='gray', linewidth=0.5)

    # Draw border directions
    margin = 0.5
    plt.axvline(env.size + margin, color='blue', linewidth=16)    # Right
    plt.axvline(0 - margin, color='yellow', linewidth=16)         # Left
    plt.axhline(env.size + margin, color='green', linewidth=16)   # Top
    plt.axhline(0 - margin, color='orange', linewidth=16)         # Bottom

    plt.gca().set_xlim([0 - margin, env.size + margin])
    plt.gca().set_ylim([0 - margin, env.size + margin])
    plt.axis('off')
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()