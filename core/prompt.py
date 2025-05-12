def build_prompt_single(agent_pos, target_pos, valid_actions, grid_size):
    action_list = ', '.join([f"**{a}**" for a in valid_actions])
    return f"""
You are looking at an 8x8 grid world that has colored borders to indicate direction:

- The **top** border is **green** — this is the **up** direction.
- The **bottom** border is **gray** — this is the **down** direction.
- The **left** border is **yellow** — this is the **left** direction.
- The **right** border is **blue** — this is the **right** direction.

Inside the grid:
- The **black square** is the agent.
- The **red square** is the goal.

Each square can be referenced by a (row, column) coordinate.
- Coordinates are zero-indexed.
- (0, 0) is the **bottom-left** corner of the grid.
- ({grid_size-1}, {grid_size-1}) is the **top-right** corner.

Current situation:
- Agent is at **(row {agent_pos[0]}, column {agent_pos[1]})**
- Goal is at **(row {target_pos[0]}, column {target_pos[1]})**

The following directions are valid from the agent's current position:
{action_list}

Your task:
Help the agent move **one step closer** to the goal, using only one of the **valid directions above**.

Respond with **one word only**: {', '.join([f'**{a}**' for a in valid_actions])} — based on the image.
"""

def build_prompt_first_agent(agent1_pos, agent2_pos, goal1_pos, valid_actions, grid_size):
    action_list = ', '.join([f"**{a}**" for a in valid_actions])
    return f"""
You are looking at an 8x8 grid world. You are Agent 1 (black square).

Inside the grid:
- You are the **black square** (Agent 1)
- There is another agent shown as a **gray square** (Agent 2)
- Your goal is the **red square**
- There is another goal shown as an **orange square** (not your goal)

Each square can be referenced by a (row, column) coordinate.
- Coordinates are zero-indexed
- (0, 0) is the bottom-left corner
- ({grid_size-1}, {grid_size-1}) is the top-right corner

Current situation:
- You (Agent 1) are at **(row {agent1_pos[0]}, column {agent1_pos[1]})**
- Agent 2 is at **(row {agent2_pos[0]}, column {agent2_pos[1]})**
- Your goal (red) is at **(row {goal1_pos[0]}, column {goal1_pos[1]})**

Valid moves from your current position:
{action_list}

Your task:
Move one step closer to YOUR goal (red square), using only valid directions.
Avoid colliding with the other agent if possible.

Respond with one word only: {', '.join([f'**{a}**' for a in valid_actions])}
"""

def build_prompt_second_agent(agent1_pos, agent2_pos, goal2_pos, valid_actions, grid_size):
    action_list = ', '.join([f"**{a}**" for a in valid_actions])
    return f"""
You are looking at an 8x8 grid world. You are Agent 2 (gray square).

Inside the grid:
- You are the **gray square** (Agent 2)
- There is another agent shown as a **black square** (Agent 1)
- Your goal is the **orange square**
- There is another goal shown as a **red square** (not your goal)

Each square can be referenced by a (row, column) coordinate.
- Coordinates are zero-indexed
- (0, 0) is the bottom-left corner
- ({grid_size-1}, {grid_size-1}) is the top-right corner

Current situation:
- You (Agent 2) are at **(row {agent2_pos[0]}, column {agent2_pos[1]})**
- Agent 1 is at **(row {agent1_pos[0]}, column {agent1_pos[1]})**
- Your goal (orange) is at **(row {goal2_pos[0]}, column {goal2_pos[1]})**

Valid moves from your current position:
{action_list}

Your task:
Move one step closer to YOUR goal (orange square), using only valid directions.
Avoid colliding with the other agent if possible.

Respond with one word only: {', '.join([f'**{a}**' for a in valid_actions])}
"""