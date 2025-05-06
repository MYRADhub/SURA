import matplotlib.pyplot as plt
import numpy as np

def plot_grid(grid_size, agent_pos, goal_pos, image_path="grid.png"):
    # Create grid and positions
    grid = np.ones((grid_size, grid_size, 3))

    # Set colors
    grid[agent_pos] = [0.5, 0.5, 0.5]  # Gray (agent)
    grid[goal_pos] = [0, 0, 0]         # Black (goal)

    # Plot and save
    plt.imshow(grid, extent=[0, grid_size, 0, grid_size], origin='lower')
    for x in range(grid_size + 1):
        plt.axhline(x, color='gray', linewidth=0.5)
        plt.axvline(x, color='gray', linewidth=0.5)
    plt.xticks([])
    plt.yticks([])

    # Add colored boundary lines with a small margin from the grid
    margin = 0.5

    # Draw colored boundary lines (do this before turning off axis)
    plt.axvline(grid_size + margin, color='blue', linewidth=16)     # Right (vertical blue)
    plt.axvline(0 - margin, color='yellow', linewidth=16)           # Left (vertical yellow)
    plt.axhline(grid_size + margin, color='green', linewidth=16)    # Top (horizontal green)
    plt.axhline(0 - margin, color='orange', linewidth=16)             # Bottom (horizontal orange)

    # Set limits so grid does not extend under the boundary lines
    plt.gca().set_xlim([0 - margin, grid_size + margin])
    plt.gca().set_ylim([0 - margin, grid_size + margin])
    plt.axis('off')

    plt.savefig(image_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    grid_size = 8
    agent_pos = (3, 4)  # Example agent position
    goal_pos = (6, 2)   # Example goal position

    plot_grid(grid_size, agent_pos, goal_pos)
    print(f"Grid saved as 'grid.png'")