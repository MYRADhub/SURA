import argparse
from core.environment import GridWorld
from core.plot import plot_grid_unassigned_labeled

def main(config_path, output_path):
    # Load environment
    env = GridWorld(config_path)
    print(f"Loaded grid of size {env.size} with:")
    print(f"- {len(env.agents)} agents")
    print(f"- {len(env.goals)} goals")
    print(f"- {len(env.obstacles)} obstacles")

    # Plot grid
    plot_grid_unassigned_labeled(env, image_path=output_path)
    print(f"Grid visualization saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a grid world YAML configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument("--output", type=str, default="data/grid_visualization.png", help="Output image file path")
    args = parser.parse_args()

    main(args.config, args.output)
