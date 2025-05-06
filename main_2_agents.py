import random
from plot_2_agents import plot_two_agents_two_goals
from agent import send_image_to_model_openai
from utils import (
    get_valid_actions, 
    move_agent, 
    build_prompt_first_agent,
    build_prompt_second_agent
)

def get_random_positions(grid_size):
    positions = random.sample(range(grid_size * grid_size), 4)
    agent1_pos = divmod(positions[0], grid_size)
    agent2_pos = divmod(positions[1], grid_size)
    goal1_pos = divmod(positions[2], grid_size)
    goal2_pos = divmod(positions[3], grid_size)
    return agent1_pos, agent2_pos, goal1_pos, goal2_pos

def extract_direction(response):
    for dir_candidate in ['up', 'down', 'left', 'right']:
        if dir_candidate in response.lower():
            return dir_candidate
    return None

if __name__ == "__main__":
    grid_size = 6
    image_path = "grid_2_agents.png"
    
    # Initialize positions
    agent1_pos, agent2_pos, goal1_pos, goal2_pos = get_random_positions(grid_size)
    init_agent1_pos = agent1_pos
    init_agent2_pos = agent2_pos
    
    step = 0
    agent1_done = False
    agent2_done = False
    collision_count = 0  # Add collision counter
    
    while not (agent1_done and agent2_done):
        print(f"\n--- Step {step} ---")
        print(f"Agent 1: {agent1_pos}, Goal 1: {goal1_pos}")
        print(f"Agent 2: {agent2_pos}, Goal 2: {goal2_pos}")
        
        # Plot current state
        plot_two_agents_two_goals(
            grid_size, agent1_pos, agent2_pos, 
            goal1_pos, goal2_pos, image_path
        )
        
        new_agent1_pos = agent1_pos
        new_agent2_pos = agent2_pos
        
        # Get moves for both agents if they haven't reached their goals
        if not agent1_done:
            valid_actions1 = get_valid_actions(agent1_pos, grid_size)
            prompt1 = build_prompt_first_agent(
                agent1_pos, agent2_pos, goal1_pos, valid_actions1, grid_size
            )
            response1 = send_image_to_model_openai(image_path, prompt1, temperature=0.0000001)
            direction1 = extract_direction(response1)
            if direction1:
                new_agent1_pos = move_agent(agent1_pos, direction1, grid_size)
        
        if not agent2_done:
            valid_actions2 = get_valid_actions(agent2_pos, grid_size)
            prompt2 = build_prompt_second_agent(
                agent1_pos, agent2_pos, goal2_pos, valid_actions2, grid_size
            )
            response2 = send_image_to_model_openai(image_path, prompt2, temperature=0.0000001)
            direction2 = extract_direction(response2)
            if direction2:
                new_agent2_pos = move_agent(agent2_pos, direction2, grid_size)
        
        # Check for collision
        if new_agent1_pos == new_agent2_pos:
            collision_count += 1  # Increment collision counter
            # Randomly choose which agent gets to move
            if random.random() < 0.5:
                new_agent2_pos = agent2_pos  # Agent 2 stays put
            else:
                new_agent1_pos = agent1_pos  # Agent 1 stays put
        
        # Update positions
        agent1_pos = new_agent1_pos
        agent2_pos = new_agent2_pos
        
        # Check if agents reached their goals
        if agent1_pos == goal1_pos:
            agent1_done = True
        if agent2_pos == goal2_pos:
            agent2_done = True
            
        step += 1
    
    print("\nTask completed!")
    # Calculate optimal paths
    optimal_path1 = abs(init_agent1_pos[0] - goal1_pos[0]) + abs(init_agent1_pos[1] - goal1_pos[1])
    optimal_path2 = abs(init_agent2_pos[0] - goal2_pos[0]) + abs(init_agent2_pos[1] - goal2_pos[1])
    print(f"Optimal path lengths: Agent 1: {optimal_path1}, Agent 2: {optimal_path2}")
    print(f"Total steps taken: {step}")
    print(f"Total collisions: {collision_count}")