import random
import yaml
import os

def generate_case(case_num, grid_size=20, min_agents=2, max_agents=6, min_obstacles=15, max_obstacles=30):
    num_agents = random.randint(min_agents, max_agents)
    num_obstacles = random.randint(min_obstacles, max_obstacles)

    all_positions = set()

    # Generate unique positions for agents
    agents = []
    while len(agents) < num_agents:
        pos = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        if pos not in all_positions:
            agents.append(list(pos))
            all_positions.add(pos)

    # Generate unique positions for goals
    goals = []
    while len(goals) < num_agents:
        pos = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        if pos not in all_positions:
            goals.append(list(pos))
            all_positions.add(pos)

    # Generate unique positions for obstacles
    obstacles = []
    while len(obstacles) < num_obstacles:
        pos = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        if pos not in all_positions:
            obstacles.append(list(pos))
            all_positions.add(pos)

    # Assemble the YAML content
    case_data = {
        'size': grid_size,
        'agents': agents,
        'goals': goals,
        'obstacles': obstacles
    }

    return case_data

def main():
    output_dir = "configs/eval"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, 101):
        case_data = generate_case(case_num=i)
        filename = os.path.join(output_dir, f"case_{i}.yaml")
        with open(filename, 'w') as f:
            yaml.dump(case_data, f, sort_keys=False, default_flow_style=None)

        print(f"Generated {filename}")

if __name__ == "__main__":
    main()
