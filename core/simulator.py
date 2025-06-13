import pandas as pd
import ast
from core.environment import GridWorld
from core.plot import plot_grid_unassigned
import argparse

def simulate_from_log(csv_path, config_path, image_path="data/sim.png"):
    # Load config-based environment
    env = GridWorld(config_path)

    # Load and preprocess the CSV
    df = pd.read_csv(csv_path)
    df["position_before"] = df["position_before"].apply(ast.literal_eval)
    df["position_after"] = df["position_after"].apply(ast.literal_eval)

    agent_ids = sorted(df["agent_id"].unique())
    grouped = df.groupby("step")
    agent_positions = env.agents[:]

    print(f"🧭 Simulation start: {len(grouped)} steps")
    print(f"Initial agent positions: {env.agents}")
    print(f"Goal positions: {env.goals}")
    print(f"Obstacles: {env.obstacles}")

    plot_grid_unassigned(env, image_path=image_path)
    print(f"🖼️  Initial grid rendered to: {image_path}")
    input("🔁 Press [ENTER] to start simulation...")

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

        env.agents = agent_positions[:]
        plot_grid_unassigned(env, image_path=image_path)
        print(f"🖼️  Grid rendered to: {image_path}")
        input("🔁 Press [ENTER] for next step...")

    print("\n✅ Simulation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate agent movements from a log CSV file.")
    parser.add_argument("csv_path", help="Path to the CSV log file")
    parser.add_argument("config_path", help="Path to the environment config file")
    parser.add_argument("--image_path", default="data/sim.png", help="Path to save rendered grid images (default: data/sim.png)")

    args = parser.parse_args()

    simulate_from_log(args.csv_path, args.config_path, args.image_path)