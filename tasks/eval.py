from core.environment import GridWorld
from core.plot import plot_grid
from core.agent import send_image_to_model_openai
from core.prompt import (
    build_prompt_single,
    build_prompt_first_agent,
    build_prompt_second_agent,
    build_prompt_single_obs,
    build_prompt_first_agent_obs,
    build_prompt_second_agent_obs,
)
from core.utils import shortest_path_length
import random
import csv
import os
import argparse

GRID_SIZE = 6
NUM_RUNS = 20
MAX_STEPS = 30
IMAGE_PATH = "data/grid.png"

def extract_direction(response):
    for d in ['up', 'down', 'left', 'right']:
        if d in response.lower():
            return d
    return None

def run_1_agent(obstacles=None):
    env = GridWorld(GRID_SIZE, obstacles=obstacles)
    env.initialize_agents_goals(num_agents=1)

    agent_pos = env.agents[0]
    goal_pos = env.goals[0]
    init_agent_pos = agent_pos
    step = 0

    while agent_pos != goal_pos and step < MAX_STEPS:
        plot_grid(env, image_path=IMAGE_PATH)
        valid_actions = env.get_valid_actions(agent_pos)

        if obstacles:
            prompt = build_prompt_single_obs(agent_pos, goal_pos, valid_actions, GRID_SIZE, obstacles)
        else:
            prompt = build_prompt_single(agent_pos, goal_pos, valid_actions, GRID_SIZE)

        response = send_image_to_model_openai(IMAGE_PATH, prompt, temperature=0.0000001)
        direction = extract_direction(response)
        if not direction:
            break

        new_pos = env.move_agent(agent_pos, direction)
        if new_pos == agent_pos:
            break

        agent_pos = new_pos
        env.agents[0] = agent_pos
        step += 1

    optimal = shortest_path_length(init_agent_pos, goal_pos, env)
    return step, optimal, (step >= MAX_STEPS)

def run_2_agents(obstacles=None):
    env = GridWorld(GRID_SIZE, obstacles=obstacles)
    env.initialize_agents_goals(num_agents=2)

    agent1_pos, agent2_pos = env.agents
    goal1_pos, goal2_pos = env.goals
    init1, init2 = agent1_pos, agent2_pos

    step = 0
    done1 = done2 = False
    deleted1 = deleted2 = False
    collisions = 0

    while not (done1 and done2) and step < MAX_STEPS:
        plot_grid(env, image_path=IMAGE_PATH)
        new1, new2 = agent1_pos, agent2_pos

        # Both agents present
        if not deleted1 and not deleted2:
            if not done1:
                valid1 = env.get_valid_actions(agent1_pos)
                prompt1 = build_prompt_first_agent_obs(agent1_pos, agent2_pos, goal1_pos, valid1, GRID_SIZE, obstacles) if obstacles \
                    else build_prompt_first_agent(agent1_pos, agent2_pos, goal1_pos, valid1, GRID_SIZE)
                resp1 = send_image_to_model_openai(IMAGE_PATH, prompt1, temperature=0.0000001)
                dir1 = extract_direction(resp1)
                if dir1:
                    new1 = env.move_agent(agent1_pos, dir1)

            if not done2:
                valid2 = env.get_valid_actions(agent2_pos)
                prompt2 = build_prompt_second_agent_obs(agent1_pos, agent2_pos, goal2_pos, valid2, GRID_SIZE, obstacles) if obstacles \
                    else build_prompt_second_agent(agent1_pos, agent2_pos, goal2_pos, valid2, GRID_SIZE)
                resp2 = send_image_to_model_openai(IMAGE_PATH, prompt2, temperature=0.0000001)
                dir2 = extract_direction(resp2)
                if dir2:
                    new2 = env.move_agent(agent2_pos, dir2)

            if new1 == new2:
                collisions += 1
                if random.random() < 0.5:
                    new2 = agent2_pos
                else:
                    new1 = agent1_pos

        # Only agent 1 remains
        elif not deleted1 and deleted2:
            if not done1:
                valid1 = env.get_valid_actions(agent1_pos)
                prompt1 = build_prompt_single_obs(agent1_pos, goal1_pos, valid1, GRID_SIZE, obstacles) if obstacles \
                    else build_prompt_single(agent1_pos, goal1_pos, valid1, GRID_SIZE)
                resp1 = send_image_to_model_openai(IMAGE_PATH, prompt1, temperature=0.0000001)
                dir1 = extract_direction(resp1)
                if dir1:
                    new1 = env.move_agent(agent1_pos, dir1)
            new2 = None

        # Only agent 2 remains
        elif deleted1 and not deleted2:
            if not done2:
                valid2 = env.get_valid_actions(agent2_pos)
                prompt2 = build_prompt_single_obs(agent2_pos, goal2_pos, valid2, GRID_SIZE, obstacles) if obstacles \
                    else build_prompt_single(agent2_pos, goal2_pos, valid2, GRID_SIZE)
                resp2 = send_image_to_model_openai(IMAGE_PATH, prompt2, temperature=0.0000001)
                dir2 = extract_direction(resp2)
                if dir2:
                    new2 = env.move_agent(agent2_pos, dir2)
            new1 = None

        # Update positions
        agent1_pos = new1 if new1 is not None else agent1_pos
        agent2_pos = new2 if new2 is not None else agent2_pos
        env.agents = [pos for pos in [agent1_pos, agent2_pos] if pos is not None]

        # Check if agents reached goals and remove them
        if not deleted1 and agent1_pos == goal1_pos:
            done1 = True
            deleted1 = True
            env.remove_agent(0)
            env.remove_goal(0)
            agent1_pos = None

        if not deleted2 and agent2_pos == goal2_pos:
            idx = 1 if not deleted1 else 0
            done2 = True
            deleted2 = True
            env.remove_agent(idx)
            env.remove_goal(idx)
            agent2_pos = None

        step += 1

    opt1 = shortest_path_length(init1, goal1_pos, env)
    opt2 = shortest_path_length(init2, goal2_pos, env)
    return step, max(opt1, opt2), (step >= MAX_STEPS), collisions

def evaluate(task_name, runner):
    print(f"\n=== Evaluating {task_name} ===")
    total_steps = total_opt = fails = total_collisions = 0
    log_file = f"results/{task_name}_results.csv"
    write_header = not os.path.exists(log_file)

    with open(log_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            if task_name.startswith("2_agents"):
                writer.writerow(["Episode", "Steps", "Optimal", "Failed", "Collisions"])
            else:
                writer.writerow(["Episode", "Steps", "Optimal", "Failed"])
        for i in range(NUM_RUNS):
            print(f"Episode {i+1}/{NUM_RUNS}...")
            result = runner()
            if task_name.startswith("2_agents"):
                steps, optimal, failed, collisions = result
                total_collisions += collisions
                writer.writerow([i+1, steps, optimal, int(failed), collisions])
            else:
                steps, optimal, failed = result
                writer.writerow([i+1, steps, optimal, int(failed)])

            total_steps += steps
            total_opt += optimal
            if failed:
                fails += 1

    avg_steps = total_steps / NUM_RUNS
    avg_opt = total_opt / NUM_RUNS
    diff = avg_steps - avg_opt
    percent = (diff / avg_opt) * 100 if avg_opt else 0

    print(f"\n== {task_name} Summary ==")
    print(f"Average optimal path: {avg_opt:.2f}")
    print(f"Average steps taken : {avg_steps:.2f}")
    print(f"Difference          : {diff:.2f} ({percent:.2f}%)")
    print(f"Failures (max steps): {fails}/{NUM_RUNS}")
    if task_name.startswith("2_agents"):
        print(f"Collisions          : {total_collisions} total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GridWorld agents.")
    parser.add_argument(
        "--agents",
        type=str,
        choices=["1_agent", "1_agent_obs", "2_agents", "2_agents_obs"],
        nargs="+",
        default=["1_agent", "1_agent_obs", "2_agents", "2_agents_obs"],
        help="Specify which agents to evaluate (space separated, e.g. --agents 1_agent 2_agents)."
    )
    args = parser.parse_args()

    agent_configs = {
        "1_agent": lambda: run_1_agent(obstacles=None),
        "1_agent_obs": lambda: run_1_agent(obstacles={(2, 2), (3, 3), (1, 4)}),
        "2_agents": lambda: run_2_agents(obstacles=None),
        "2_agents_obs": lambda: run_2_agents(obstacles={(1, 1), (2, 3), (4, 2), (3, 4)}),
    }

    for agent in args.agents:
        evaluate(agent, agent_configs[agent])
