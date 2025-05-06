import matplotlib.pyplot as plt
import numpy as np

def plot_two_agents_two_goals(grid_size, agent1_pos, agent2_pos, goal1_pos, goal2_pos, image_path="grid_2_agents.png"):
    # Create a blank white grid
    grid = np.ones((grid_size, grid_size, 3))

    # Set colors
    grid[agent1_pos] = [0, 0, 0]       # Black (agent 1)
    grid[agent2_pos] = [0.5, 0.5, 0.5] # Gray (agent 2)
    grid[goal1_pos] = [1, 0, 0]        # Red (goal 1)
    grid[goal2_pos] = [1, 0.5, 0]      # Orange (goal 2)

    # Plot the grid
    plt.imshow(grid, extent=[0, grid_size, 0, grid_size], origin='lower')

    # Draw grid lines
    for x in range(grid_size + 1):
        plt.axhline(x, color='gray', linewidth=0.5)
        plt.axvline(x, color='gray', linewidth=0.5)

    # Add labels for agents and goals
    # plt.text(agent1_pos[1] + 0.5, agent1_pos[0] + 0.5, 'A1', color='white',
    #          ha='center', va='center', fontsize=12, weight='bold')
    # plt.text(agent2_pos[1] + 0.5, agent2_pos[0] + 0.5, 'A2', color='black',
    #          ha='center', va='center', fontsize=12, weight='bold')
    # plt.text(goal1_pos[1] + 0.5, goal1_pos[0] + 0.5, 'G1', color='white',
    #          ha='center', va='center', fontsize=12, weight='bold')
    # plt.text(goal2_pos[1] + 0.5, goal2_pos[0] + 0.5, 'G2', color='black',
    #          ha='center', va='center', fontsize=12, weight='bold')

    # Add colored border lines
    margin = 0.5
    plt.axvline(grid_size + margin, color='blue', linewidth=16)     # Right
    plt.axvline(0 - margin, color='yellow', linewidth=16)           # Left
    plt.axhline(grid_size + margin, color='green', linewidth=16)    # Top
    plt.axhline(0 - margin, color='orange', linewidth=16)           # Bottom

    plt.gca().set_xlim([0 - margin, grid_size + margin])
    plt.gca().set_ylim([0 - margin, grid_size + margin])
    plt.axis('off')

    plt.savefig(image_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    grid_size = 8
    agent1_pos = (2, 3)
    agent2_pos = (5, 5)
    goal1_pos = (7, 1)
    goal2_pos = (0, 6)

    plot_two_agents_two_goals(grid_size, agent1_pos, agent2_pos, goal1_pos, goal2_pos)
    print("Grid with 2 agents and 2 goals saved as 'grid_2agents.png'")
