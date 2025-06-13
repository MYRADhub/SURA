import yaml
from core.environment import GridWorld

def load_env_from_yaml(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    env = GridWorld(size=config['size'])
    env.obstacles = set(tuple(o) for o in config.get('obstacles', []))
    env.agents = [tuple(a) if a else None for a in config.get('agents', [])]
    env.goals = [tuple(g) if g else None for g in config.get('goals', [])]
    return env
