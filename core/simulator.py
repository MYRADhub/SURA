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

    # Get agent IDs from the DataFrame
    agent_ids = sorted(df["agent_id"].unique())

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

    # Render initial state before any moves
    plot_grid_unassigned(env, image_path=image_path)
    print(f"🖼️  Initial grid rendered to: {image_path}")
    input("🔁 Press [ENTER] to start simulation...")

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
    agent_starts=[(2, 0), (2, 3)]
    goal_positions=[(1, 4), (7, 7)]
    obstacles={(3, 3), (4, 4), (2, 5), (5, 2), (6, 6)}
    simulate_from_log(csv_path, agent_starts, goal_positions, obstacles=obstacles)
