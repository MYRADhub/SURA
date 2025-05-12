import matplotlib.pyplot as plt
import numpy as np
from core.environment import GridWorld

def plot_grid(env: GridWorld, image_path="grid_2_agents.png"):
    grid = np.ones((env.size, env.size, 3))

    for obs in env.obstacles:
        grid[obs] = [0.4, 0.2, 0]  # Brown

    if len(env.agents) > 0:
        grid[env.agents[0]] = [0, 0, 0]       # Black
    if len(env.agents) > 1:
        grid[env.agents[1]] = [0.5, 0.5, 0.5] # Gray

    if len(env.goals) > 0:
        grid[env.goals[0]] = [1, 0, 0]        # Red
    if len(env.goals) > 1:
        grid[env.goals[1]] = [1, 0.5, 0]      # Orange

    plt.imshow(grid, extent=[0, env.size, 0, env.size], origin='lower')
    for x in range(env.size + 1):
        plt.axhline(x, color='gray', linewidth=0.5)
        plt.axvline(x, color='gray', linewidth=0.5)

    margin = 0.5
    plt.axvline(env.size + margin, color='blue', linewidth=16)
    plt.axvline(0 - margin, color='yellow', linewidth=16)
    plt.axhline(env.size + margin, color='green', linewidth=16)
    plt.axhline(0 - margin, color='orange', linewidth=16)

    plt.gca().set_xlim([0 - margin, env.size + margin])
    plt.gca().set_ylim([0 - margin, env.size + margin])
    plt.axis('off')
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    env = GridWorld(size=8, obstacles={(3, 3), (1, 4), (5, 2)})
    env.initialize_agents_goals(num_agents=2)
    plot_grid(env)
    print("Grid with 2 agents and obstacles saved.")
