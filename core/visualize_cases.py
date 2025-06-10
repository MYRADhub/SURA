from core.environment import GridWorld
from core.plot import plot_grid_unassigned

# Shared obstacles for all test cases
obstacles = {
    (3, 3),
    (4, 4),
    (2, 5),
    (5, 2),
    (6, 6),
}


# Define each test case with custom agent and goal coordinates
cases = {
    "case_1_local_greed": {
        "agents": [(3, 1), (3, 5)],
        "goals":  [(1, 4), (0, 6)],
    },
    "case_2_decoy_efficiency": {
        "agents": [(0, 0), (7, 0)],
        "goals":  [(7, 7), (3, 4)],
    },
    "case_3_one_stays_back": {
        "agents": [(1, 1), (2, 1), (7, 6)],
        "goals":  [(6, 7), (1, 7), (3, 6)],
    },
    "case_4_tight_corridor": {
        "agents": [(0, 1), (1, 0), (0, 0)],
        "goals":  [(6, 1), (7, 2), (7, 1)],
    },
    "case_5_decoy_ignored": {
        "agents": [(7, 0), (6, 0), (0, 7)],
        "goals":  [(0, 0), (1, 1), (2, 2)],
    },
    "case_6_lane_assignment": {
        "agents": [(2, 2), (2, 3), (2, 4)],
        "goals":  [(6, 2), (6, 3), (6, 4)],
    },
    "case_7_vertical_choke": {
        "agents": [(0, 3), (0, 5), (0, 7)],
        "goals":  [(7, 3), (7, 5), (7, 7)],
    },
    "case_8_goal_stealing": {
        "agents": [(6, 0), (6, 1), (6, 2)],
        "goals":  [(0, 0), (0, 1), (0, 2)],
    },
}

# Generate and save a plot for each case
for case_name, config in cases.items():
    env = GridWorld(size=8, obstacles=obstacles)
    env.initialize_agents_goals_custom(agents=config["agents"], goals=config["goals"])
    path = f"data/{case_name}.png"
    plot_grid_unassigned(env, image_path=path)
    print(f"✅ Saved {path}")
