import pandas as pd
import ast
from core.environment import GridWorld
from core.plot import plot_grid_unassigned

def simulate_from_log(
    csv_path,
    initial_positions,
    goal_positions,
    grid_size=8,
    obstacles={(2, 2), (3, 3), (4, 1)},
    image_path="data/sim.png"
):
    # Load and preprocess the CSV
    df = pd.read_csv(csv_path)
    df["position_before"] = df["position_before"].apply(ast.literal_eval)
    df["position_after"] = df["position_after"].apply(ast.literal_eval)

    # Get initial agent positions and goal declarations
    first_step = df[df["step"] == 0]
    agent_ids = sorted(first_step["agent_id"].unique())
    num_agents = len(agent_ids)

    # Deduce goal declarations per agent from first non-empty target_goal
    goal_declarations = [None] * num_agents
    for i, aid in enumerate(agent_ids):
        agent_rows = df[df["agent_id"] == aid]
        for g in agent_rows["target_goal"]:
            if pd.notna(g) and isinstance(g, str) and g.strip() != "":
                goal_declarations[i] = g.strip().upper()
                break

    # Initialize the environment
    env = GridWorld(size=grid_size, obstacles=obstacles)
    env.initialize_agents_goals_custom(agents=initial_positions, goals=goal_positions)

    # Group by step
    grouped = df.groupby("step")
    agent_positions = initial_positions[:]

    print(f"🧭 Simulation start: {len(grouped)} steps")
    print(f"Initial agent positions: {agent_positions}")
    print(f"Goal positions: {goal_positions}")
    print(f"Obstacles: {obstacles}")
    print(f"Press ENTER to advance each step...")

    for step, rows in grouped:
        print(f"\n📦 STEP {step}")
        for _, row in rows.iterrows():
            aid = int(row["agent_id"])
            idx = agent_ids.index(aid)
            before = row["position_before"]
            after = row["position_after"]
            dir = row["chosen_direction"]
            yes_logprob = row["logprob_yes"]
            target = row["target_goal"]
            top1 = row["goal_top1"]
            top1p = row["goal_top1_logprob"]
            top2 = row["goal_top2"]
            top2p = row["goal_top2_logprob"]
            explanation = row["explanation"]

            print(f"→ Agent {aid} moves {dir.upper()} from {before} to {after}")
            print(f"   Target goal: {target}")
            print(f"   LogProb YES: {yes_logprob}")
            print(f"   Top goals: {top1} ({top1p}), {top2} ({top2p})")
            print(f"   Reason: {explanation}")

            agent_positions[idx] = after

        # Update and render environment
        env.agents = agent_positions[:]
        plot_grid_unassigned(env, image_path=image_path)
        print(f"🖼️  Grid rendered to: {image_path}")
        input("🔁 Press [ENTER] for next step...")

    print("\n✅ Simulation complete.")

if __name__ == "__main__":
    csv_path = "data/agent_step_logs.csv"
    initial_positions = [(0, 0), (0, 1), (0, 2)]
    goal_positions = [(7, 7), (7, 6), (7, 5)]
    simulate_from_log(csv_path, initial_positions, goal_positions)
