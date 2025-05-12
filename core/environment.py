import random

class GridWorld:
    def __init__(self, size, obstacles=None):
        self.size = size
        self.obstacles = set(obstacles or [])

        # Init empty agents and goals
        self.agents = []   # List of (row, col)
        self.goals = []    # List of (row, col)

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

