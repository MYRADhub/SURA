import random
import yaml
from collections import deque
from core.utils import shortest_path_length

class GridWorld:
    def __init__(self, size_or_config, obstacles=None):
        if isinstance(size_or_config, str):
            # Assume it's a path to a YAML config file
            self._load_from_config(size_or_config)
        else:
            # Manual size + optional obstacles mode
            self.size = size_or_config
            self.obstacles = set(obstacles or [])
            self.agents = []
            self.goals = []

    def _load_from_config(self, path):
        try:
            with open(path, "r") as f:
                cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in config file: {e}")

        if not isinstance(cfg, dict) or "size" not in cfg:
            raise ValueError("Config file missing required 'size' field or is not a valid dictionary.")

        self.size = cfg["size"]
        self.obstacles = {tuple(pos) for pos in cfg.get("obstacles", [])}
        self.agents = [tuple(pos) for pos in cfg.get("agents", [])]
        self.goals = [tuple(pos) for pos in cfg.get("goals", [])]

    def is_valid(self, pos):
        row, col = pos
        return (
            0 <= row < self.size and
            0 <= col < self.size and
            pos not in self.obstacles and
            pos not in self.agents
        )

    def sample_position(self, exclude=None):
        exclude = set(exclude or [])
        candidates = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.is_valid((r, c)) and (r, c) not in exclude
        ]
        return random.choice(candidates) if candidates else None

    def initialize_agents_goals(self, num_agents=1):
        self.agents = []
        self.goals = []

        used = set(self.obstacles)

        for _ in range(num_agents):
            agent = self.sample_position(used)
            used.add(agent)
            self.agents.append(agent)

        for _ in range(num_agents):
            goal = self.sample_position(used)
            used.add(goal)
            self.goals.append(goal)

    def initialize_agents_goals_custom(self, agents, goals):
        """
        Initialize agents and goals with provided positions.
        Args:
            agents: list of (row, col) tuples for agent positions
            goals: list of (row, col) tuples for goal positions
        Raises:
            ValueError if any position is invalid or overlapping
        """
        if len(agents) != len(goals):
            raise ValueError("Number of agents and goals must match.")

        used = set(self.obstacles)
        for pos in agents + goals:
            if pos in used or not self.is_valid(pos):
                raise ValueError(f"Invalid or overlapping position: {pos}")
            used.add(pos)

        self.agents = list(agents)
        self.goals = list(goals)

    def get_valid_actions(self, agent_pos):
        row, col = agent_pos
        actions = []
        candidates = {
            "up": (row + 1, col),
            "down": (row - 1, col),
            "left": (row, col - 1),
            "right": (row, col + 1),
        }
        for direction, new_pos in candidates.items():
            if self.is_valid(new_pos):
                actions.append(direction)
        return actions

    def move_agent(self, agent_pos, direction):
        row, col = agent_pos
        if direction == 'up':
            new_pos = (row + 1, col)
        elif direction == 'down':
            new_pos = (row - 1, col)
        elif direction == 'left':
            new_pos = (row, col - 1)
        elif direction == 'right':
            new_pos = (row, col + 1)
        else:
            return agent_pos  # Invalid direction

        return new_pos if self.is_valid(new_pos) else agent_pos

    def remove_agent(self, index):
        if 0 <= index < len(self.agents):
            self.agents.pop(index)
        else:
            raise IndexError("Agent index out of range.")
        
    def remove_goal(self, index):
        if 0 <= index < len(self.goals):
            self.goals.pop(index)
        else:
            raise IndexError("Goal index out of range.")

    def assignment_cost(self, assignment: dict[int, str]) -> int:
        """
        Compute the maximum BFS path length (cost) of a given agent-to-goal assignment.

        Args:
            assignment (dict[int, str]): Mapping from agent ID (1-based) to goal letter ('A', 'B', ...)

        Returns:
            int: The maximum path length among all agent-goal pairs (team cost).
        """
        costs = []

        for agent_id, goal_letter in assignment.items():
            agent_idx = agent_id - 1
            goal_idx = ord(goal_letter.upper()) - ord('A')

            if agent_idx >= len(self.agents):
                raise IndexError(f"Agent {agent_id} is out of bounds.")
            if goal_idx >= len(self.goals):
                raise IndexError(f"Goal '{goal_letter}' is out of bounds.")

            agent_pos = self.agents[agent_idx]
            goal_pos = self.goals[goal_idx]

            if agent_pos is None or goal_pos is None:
                costs.append(float('inf'))
            else:
                dist = shortest_path_length(agent_pos, goal_pos, self)
                costs.append(dist)

        return max(costs) if costs else float('inf')
