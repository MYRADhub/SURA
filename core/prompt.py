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

def build_prompt_single_obs(agent_pos, goal_pos, valid_actions, grid_size, obstacles):
    action_list = ', '.join([f"{a}" for a in valid_actions])
    obs_coords = ', '.join([f"({r}, {c})" for r, c in sorted(obstacles)])

    return f"""
You are looking at an {grid_size}x{grid_size} grid world that has colored borders to indicate direction:

- The **top** border is **green** — this is the **up** direction.
- The **bottom** border is **gray** — this is the **down** direction.
- The **left** border is **yellow** — this is the **left** direction.
- The **right** border is **blue** — this is the **right** direction.

Inside the grid:
- The **black square** is the agent.
- The **red square** is the goal.
- **Brown squares** represent obstacles that cannot be entered.

Coordinates are zero-indexed.
- (0, 0) is the bottom-left corner
- ({grid_size - 1}, {grid_size - 1}) is the top-right corner

Current situation:
- Agent is at **(row {agent_pos[0]}, column {agent_pos[1]})**
- Goal is at **(row {goal_pos[0]}, column {goal_pos[1]})**
- Obstacles are at: {obs_coords}

Valid directions from the agent's current position:
{action_list}
You can only move one step in one of these directions.

Your task:
Help the agent move **one step closer** to the goal while avoiding obstacles.
Only choose from the valid directions listed.

Respond with **one word only**: {', '.join([f'{a}' for a in valid_actions])}

MAKE SURE to respond with one word only, CHOSEN FROM {action_list}, all lowercase, not bolded, not capitalized, and without any extra context or explanation.
"""

def build_prompt_first_agent_obs(agent1_pos, agent2_pos, goal1_pos, valid_actions, grid_size, obstacles):
    action_list = ', '.join([f"**{a}**" for a in valid_actions])
    obs_coords = ', '.join([f"({r}, {c})" for r, c in sorted(obstacles)])

    return f"""
You are Agent 1 (the **black square**) in an {grid_size}x{grid_size} grid world.

Grid orientation:
- **Green** top border = up
- **Gray** bottom border = down
- **Yellow** left border = left
- **Blue** right border = right

Inside the grid:
- **You** are the **black square**
- **Agent 2** is the **gray square**
- **Your goal** is the **red square**
- **Another goal** (not yours) is the **orange square**
- **Brown squares** are **obstacles** that block movement

Coordinates are zero-indexed:
- (0, 0) = bottom-left
- ({grid_size - 1}, {grid_size - 1}) = top-right

State:
- You are at **(row {agent1_pos[0]}, column {agent1_pos[1]})**
- Agent 2 is at **(row {agent2_pos[0]}, column {agent2_pos[1]})**
- Your goal is at **(row {goal1_pos[0]}, column {goal1_pos[1]})**
- Obstacles are at: {obs_coords}

Valid directions for you:
{action_list}

Your task:
Move one step toward your goal while avoiding obstacles and other agents.

Respond with **one word only**: {', '.join([f'**{a}**' for a in valid_actions])}
"""

def build_prompt_single_obs_v2(
    agent_pos,
    goal_pos,
    valid_actions,
    grid_size,
    obstacles,
    memory, # e.g. [(1, 3, "up", 2, 3), ...]
    visits # e.g. {(2,3): 2, (1,2): 1, ...}
):
    action_list = ", ".join([f"{a}" for a in valid_actions])

    # Memory section
    if memory:
        history_lines = "\n".join(
            [f"  • {i+1}. you moved from (row {x0}, col {y0}) **{dir_}** → got to (row {x1}, col {y1})"
             for i, (x0, y0, dir_, x1, y1) in enumerate(memory[:5])]
        )
    else:
        history_lines = "  • (no prior moves — this is the first step)"

    # Obstacle string
    obs_coords = ", ".join([f"({r}, {c})" for r, c in sorted(obstacles)])

    # Move analysis
    move_analysis = []
    for direction in valid_actions:
        # Determine resulting cell
        r, c = agent_pos
        if direction == "up":
            target = (r + 1, c)
        elif direction == "down":
            target = (r - 1, c)
        elif direction == "left":
            target = (r, c - 1)
        elif direction == "right":
            target = (r, c + 1)
        else:
            continue  # skip invalid direction (just in case)

        count = visits.get(target, 0)
        move_analysis.append(f"  • {direction:5} → (row {target[0]}, col {target[1]}) — visited {count} time(s)")

    move_analysis_block = "\n".join(move_analysis)

    return f"""
**Environment**

You are controlling a single black square (the *agent*) on a {grid_size}×{grid_size} grid.
Four thick coloured borders indicate global orientation:

* green (top) → **up**
* gray (bottom) → **down**
* yellow (left) → **left**
* blue (right) → **right**

Inside the grid each cell is referenced by **zero-indexed coordinates** — (0 , 0) is the bottom-left corner, ({grid_size-1} , {grid_size-1}) the top-right.
A red square marks the goal; brown squares are immovable obstacles the agent **cannot** enter.

Current state  
* agent position … **(row {agent_pos[0]}, col {agent_pos[1]})**  
* goal   position … **(row {goal_pos[0]}, col {goal_pos[1]})**  
* obstacle cells  … {obs_coords or "none"}

**Memory (last 5 moves)**  
{history_lines}

**Move Analysis (cell visit frequency)**  
{move_analysis_block}

---

### Instructions (think silently, output nothing but the chosen word)

1. **Legal moves** – from your current square you may move exactly one step in any of these directions: {action_list}.  
   Trying to step into an obstacle leaves you in the same place.
2. **Primary objective** – pick the move that *reduces the Manhattan distance* to the red goal whenever possible.
3. **Look-ahead** – mentally consider the next one-to-two steps to avoid dead-ends or traps.
4. **No blind repetition** – avoid repeating the previous direction unless it clearly improves progress.
5. **Obstacle awareness** – never select a direction that collides with a brown square.
6. **Finish rule** – when the agent reaches the goal coordinate, no further moves are required.
7. **Output format** – respond with **one lowercase word only** ({action_list}).  
   *Do not add punctuation, boldface, extra spaces, or any extra explanation.*

Now choose the best move.
"""



def build_prompt_second_agent_obs(agent1_pos, agent2_pos, goal2_pos, valid_actions, grid_size, obstacles):
    action_list = ', '.join([f"**{a}**" for a in valid_actions])
    obs_coords = ', '.join([f"({r}, {c})" for r, c in sorted(obstacles)])

    return f"""
You are Agent 2 (the **gray square**) in an {grid_size}x{grid_size} grid world.

Grid orientation:
- **Green** top border = up
- **Gray** bottom border = down
- **Yellow** left border = left
- **Blue** right border = right

Inside the grid:
- **You** are the **gray square**
- **Agent 1** is the **black square**
- **Your goal** is the **orange square**
- **Another goal** (not yours) is the **red square**
- **Brown squares** are **obstacles** that block movement

Coordinates are zero-indexed:
- (0, 0) = bottom-left
- ({grid_size - 1}, {grid_size - 1}) = top-right

State:
- You are at **(row {agent2_pos[0]}, column {agent2_pos[1]})**
- Agent 1 is at **(row {agent1_pos[0]}, column {agent1_pos[1]})**
- Your goal is at **(row {goal2_pos[0]}, column {goal2_pos[1]})**
- Obstacles are at: {obs_coords}

Valid directions for you:
{action_list}

Your task:
Move one step toward your goal while avoiding obstacles and other agents.

Respond with **one word only**: {', '.join([f'**{a}**' for a in valid_actions])}
"""



# now we will build prompts for UCT graph construction by using log probs as weights

# first make 4 prompts for the 4 directions with yes and no

def build_yesno_prompt_single_obs(agent_pos, goal_pos, grid_size, obstacles, direction):
    obs_coords = ', '.join([f"({r}, {c})" for r, c in sorted(obstacles)])

    return f"""
You are looking at a {grid_size}x{grid_size} grid world with colored borders to indicate direction:

- Green top border → **up**
- Gray bottom border → **down**
- Yellow left border → **left**
- Blue right border → **right**

Inside the grid:
- The **black square** is the agent.
- The **red square** is the goal.
- **Brown squares** are obstacles — they block movement.

Coordinates are zero-indexed:
- (0, 0) is the bottom-left
- ({grid_size - 1}, {grid_size - 1}) is the top-right

Current state:
- Agent position: **(row {agent_pos[0]}, column {agent_pos[1]})**
- Goal position: **(row {goal_pos[0]}, column {goal_pos[1]})**
- Obstacles: {obs_coords}

Action under consideration:
- Should the agent move **{direction}**?

Only respond with **YES** or **NO**.
"""

# second make one prompt for the 4 directions

def build_multichoice_prompt_single_obs(agent_pos, goal_pos, valid_actions, grid_size, obstacles):
    action_list = ', '.join([f"**{a}**" for a in valid_actions])
    obs_coords = ', '.join([f"({r}, {c})" for r, c in sorted(obstacles)])

    return f"""
You are looking at a {grid_size}x{grid_size} grid world with colored borders to indicate direction:

- Green top border → **up**
- Gray bottom border → **down**
- Yellow left border → **left**
- Blue right border → **right**

Inside the grid:
- The **black square** is the agent.
- The **red square** is the goal.
- **Brown squares** are obstacles — they block movement.

Coordinates are zero-indexed:
- (0, 0) is the bottom-left
- ({grid_size - 1}, {grid_size - 1}) is the top-right

Current state:
- Agent is at: **(row {agent_pos[0]}, column {agent_pos[1]})**
- Goal is at: **(row {goal_pos[0]}, column {goal_pos[1]})**
- Obstacles are at: {obs_coords}

Your task:
Select the best direction to move that brings the agent closer to the goal and avoids obstacles.

Valid directions are:
{action_list}

Respond with one word only — the chosen direction.
"""
