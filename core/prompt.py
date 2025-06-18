
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
- The **blue square** is the agent.
- The **red square** is the goal.
- **Black squares** represent obstacles that cannot be entered.
- Each cell is labeled with its **(row, column)** coordinate for easy reference.
- The rows and columns are also labeled numerically on the grid image itself to help you orient the space.

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

def build_prompt_single_obs_v2(
    agent_pos,
    goal_pos,
    valid_actions,
    grid_size,
    obstacles,
    memory,
    visits
):
    action_list = ", ".join([f"{a}" for a in valid_actions])

    # Memory
    if memory:
        history_lines = "\n".join(
            [f"  • {i+1}. you moved from (row {x0}, col {y0}) **{dir_}** → got to (row {x1}, col {y1})"
             for i, (x0, y0, dir_, x1, y1) in enumerate(memory[:5])]
        )
    else:
        history_lines = "  • (no prior moves — this is the first step)"

    # Obstacles
    obs_coords = ", ".join([f"({r}, {c})" for r, c in sorted(obstacles)])

    # Move analysis
    move_analysis = []
    for direction in valid_actions:
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
            continue
        count = visits.get(target, 0)
        move_analysis.append(f"  • {direction:5} → (row {target[0]}, col {target[1]}) — visited {count} time(s)")
    move_analysis_block = "\n".join(move_analysis)

    return f"""
**Environment**

You are controlling a single blue square (the *agent*) on a {grid_size}×{grid_size} grid.
Four thick coloured borders indicate global orientation:

* green (top) → **up**
* gray (bottom) → **down**
* yellow (left) → **left**
* blue (right) → **right**

Inside the grid:
* The **blue square** labeled **A1** is the agent you control  
* The **red square** labeled **G1** is the goal you need to reach  
* **Black squares** are immovable obstacles and cannot be entered , they are labeled **O** 
* Each grid cell includes labeled **row and column indices** to help with position reference  
* The top and left edges of the image contain axis annotations for **rows** and **columns**

All coordinates are zero-indexed:
- (0, 0) is the bottom-left corner
- ({grid_size - 1}, {grid_size - 1}) is the top-right corner

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

1. **Legal moves** - from your current square you may move exactly one step in any of these directions: {action_list}.  
   Trying to step into an obstacle leaves you in the same place.
2. **Primary objective** - pick the move that *reduces the Manhattan distance* to the red goal whenever possible.
3. **Look-ahead** - mentally consider the next one-to-two steps to avoid dead-ends or traps.
4. **No blind repetition** - avoid repeating the previous direction unless it clearly improves progress.
5. **Obstacle awareness** - never select a direction that collides with a black square.
6. **Finish rule** - when the agent reaches the goal coordinate, no further moves are required.
7. **Output format** - respond with **one lowercase word only** ({action_list}).  
   *Do not add punctuation, boldface, extra spaces, or any extra explanation.*

**Important:** If two black obstacle squares touch at the corners (diagonally), a thick black line will connect them.
This indicates that the agent **cannot pass diagonally** between those cells.

You must treat this diagonal as a wall — only cardinal (up/down/left/right) movements are allowed, and diagonal movement is not possible under any condition.

Now choose the best move.
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

def build_yesno_prompt_single_obs_v2(
    agent_pos,
    goal_pos,
    grid_size,
    obstacles,
    direction,
    memory,  # list of (r0, c0, dir, r1, c1)
    visits   # dict {(r, c): count}
):
    # Format obstacle list
    obs_coords = ', '.join([f"({r}, {c})" for r, c in sorted(obstacles)])

    # Format memory (up to 5 recent moves)
    if memory:
        history_lines = "\n".join(
            [f"  • {i+1}. you moved from (row {r0}, col {c0}) **{dir_}** → got to (row {r1}, col {c1})"
             for i, (r0, c0, dir_, r1, c1) in enumerate(memory[:5])]
        )
    else:
        history_lines = "  • (no prior moves — this is the first step)"

    # Format move analysis
    move_analysis_lines = []
    for d in ['up', 'down', 'left', 'right']:
        r, c = agent_pos
        if d == "up":
            target = (r + 1, c)
        elif d == "down":
            target = (r - 1, c)
        elif d == "left":
            target = (r, c - 1)
        elif d == "right":
            target = (r, c + 1)
        else:
            continue
        count = visits.get(target, 0)
        move_analysis_lines.append(f"  • {d:5} → (row {target[0]}, col {target[1]}) — visited {count} time(s)")
    move_analysis = "\n".join(move_analysis_lines)

    return f"""
**Environment**

You are controlling a blue square (the agent labeled **A1**) on a {grid_size}×{grid_size} grid.  
A red square labeled **G1** marks the goal.  
Black squares are obstacles that **cannot be entered**.

Four colored borders define direction:
* green (top) → **up**
* gray (bottom) → **down**
* yellow (left) → **left**
* blue (right) → **right**

All coordinates are zero-indexed:
- (0, 0) is the bottom-left corner
- ({grid_size - 1}, {grid_size - 1}) is the top-right corner

In the image:
- Each cell is labeled with its row and column index
- Obstacles are black squares
- Your agent is labeled A1
- The goal is labeled G1

**Current state**  
* Agent position … **(row {agent_pos[0]}, col {agent_pos[1]})**  
* Goal position  … **(row {goal_pos[0]}, col {goal_pos[1]})**  
* Obstacles      … {obs_coords or "none"}

**Memory (last 5 moves)**  
{history_lines}

**Move Analysis (cell visit frequency)**  
{move_analysis}

---

### Question

Should the agent move **{direction}**?

---

### Instructions (read carefully before responding)

1. **Legal actions** - consider if this direction avoids obstacles and is allowed from the current position.
2. **Goal-seeking** - prioritize moving toward the red goal.
3. **Avoid repetition** - if the same move has been repeated without progress, say NO unless it clearly helps.
4. **Trap avoidance** - avoid directions that lead to dead ends or repeated loops.
5. **Output format** - respond with exactly one word: YES or NO. Uppercase, no punctuation, no extra text, not bolded.
   *Do not include any other explanation, characters, or formatting.*

**Important:** If two black obstacle squares touch at the corners (diagonally), a thick black line will connect them.
This indicates that the agent **cannot pass diagonally** between those cells.

You must treat this diagonal as a wall — only cardinal (up/down/left/right) movements are allowed, and diagonal movement is not possible under any condition.

Now respond: YES or NO
"""

def build_yesno_prompt_multiagent(
    agent_id,
    agent_pos,
    goal_pos,
    other_agents,  # list of tuples (other_agent_id, position)
    grid_size,
    obstacles,
    direction,
    memory,   # list of (r0, c0, dir, r1, c1)
    visits    # dict {(r, c): count}
):
    # Format obstacle coordinates
    obs_coords = ', '.join([f"({r}, {c})" for r, c in sorted(obstacles)])

    # Format memory
    if memory:
        history_lines = "\n".join(
            [f"  • {i+1}. you moved from (row {r0}, col {c0}) **{dir_}** → got to (row {r1}, col {c1})"
             for i, (r0, c0, dir_, r1, c1) in enumerate(memory[:5])]
        )
    else:
        history_lines = "  • (no prior moves — this is the first step)"

    # Format move analysis
    move_analysis_lines = []
    for d in ['up', 'down', 'left', 'right']:
        r, c = agent_pos
        if d == "up":
            target = (r + 1, c)
        elif d == "down":
            target = (r - 1, c)
        elif d == "left":
            target = (r, c - 1)
        elif d == "right":
            target = (r, c + 1)
        else:
            continue
        count = visits.get(target, 0)
        move_analysis_lines.append(f"  • {d:5} → (row {target[0]}, col {target[1]}) — visited {count} time(s)")
    move_analysis = "\n".join(move_analysis_lines)

    # Format other agents and their positions
    if other_agents:
        other_agent_lines = "\n".join(
            [f"  • Agent A{aid} is at (row {pos[0]}, col {pos[1]})" for aid, pos in other_agents]
        )
    else:
        other_agent_lines = "  • (no other agents present)"

    return f"""
**Environment**

You are Agent **A{agent_id}** (a blue square labeled **A{agent_id}**) on a {grid_size}×{grid_size} grid.  
Your goal is a red square labeled **G{agent_id}**.  
Black squares are obstacles that **cannot be entered**.  
Other agents may be present — they are shown as blue squares with labels (A2, A3, etc.).  
Their goals are also marked with red squares labeled accordingly (G2, G3, ...).

Four colored borders define direction:
* green (top) → **up**
* gray (bottom) → **down**
* yellow (left) → **left**
* blue (right) → **right**

All coordinates are zero-indexed:
- (0, 0) is the bottom-left corner
- ({grid_size - 1}, {grid_size - 1}) is the top-right corner

In the image:
- Each cell is labeled with its row and column index
- Obstacles are black squares labeled **O**
- You are labeled **A{agent_id}**
- Your goal is labeled **G{agent_id}**
- Other agents are labeled A2, A3, etc.
- Other goals are labeled G2, G3, etc.

**Current state**  
* Your position         … **(row {agent_pos[0]}, col {agent_pos[1]})**  
* Your goal             … **(row {goal_pos[0]}, col {goal_pos[1]})**  
* Obstacles             … {obs_coords or "none"}  
* Other agents          …  
{other_agent_lines}

**Memory (last 5 moves)**  
{history_lines}

**Move Analysis (cell visit frequency)**  
{move_analysis}

---

### Question

Should Agent A{agent_id} move **{direction}**?

---

### Instructions (read carefully before responding)

1. **Legal actions** - determine if this direction is valid and avoids obstacles.
2. **Goal-seeking** - prioritize reducing the distance to your goal **G{agent_id}**.
3. **Avoid repetition** - if this move was repeated without progress, say NO unless clearly helpful.
4. **Trap avoidance** - avoid moves that cause loops or dead ends.
5. **Collision avoidance** - do not move into a cell currently occupied by other agents.
6. **Output format** - respond with exactly one word: YES or NO. Uppercase only, no punctuation or explanation.
7. **Diagonal rule** - if two obstacles touch diagonally, a thick black line indicates you cannot pass through that corner.
   Treat these diagonals as walls. Only up/down/left/right movement is allowed.

Now respond: YES or NO
"""

def build_yesno_code_prompt_single(
    agent_pos,
    goal_pos,
    grid_size,
    obstacles,
    direction,
    memory,  # list of (r0, c0, dir, r1, c1)
    visits   # dict {(r, c): count}
):
    # Format obstacle code lines
    obstacle_lines = "\n".join([f"obstacles.add(({r}, {c}))" for r, c in sorted(obstacles)])

    # Format memory (up to 5 recent moves)
    if memory:
        history_lines = "\n".join(
            [f"  • {i+1}. you moved from (row {r0}, col {c0}) **{dir_}** → got to (row {r1}, col {c1})"
             for i, (r0, c0, dir_, r1, c1) in enumerate(memory[:5])]
        )
    else:
        history_lines = "  • (no prior moves — this is the first step)"

    # Format move analysis
    move_analysis_lines = []
    for d in ['up', 'down', 'left', 'right']:
        r, c = agent_pos
        if d == "up":
            target = (r + 1, c)
        elif d == "down":
            target = (r - 1, c)
        elif d == "left":
            target = (r, c - 1)
        elif d == "right":
            target = (r, c + 1)
        else:
            continue
        count = visits.get(target, 0)
        move_analysis_lines.append(f"  • {d:5} → (row {target[0]}, col {target[1]}) — visited {count} time(s)")
    move_analysis = "\n".join(move_analysis_lines)

    return f"""
**Environment**

Below is a Python-style setup of the GridWorld environment.  
This code defines the grid, the agent (A1), the goal (G1), and obstacles.  
Only cardinal moves (up, down, left, right) are allowed — no diagonals.

```python
grid_size = {grid_size}

agent_pos = {agent_pos}
goal_pos = {goal_pos}

obstacles = set()
{obstacle_lines}
```

You are currently at `agent_pos`, and your goal is to reach `goal_pos`.

You will be shown an image of the current state. The image corresponds exactly to the variables above.

Your agent is labeled **A1**, the goal is **G1**, and obstacles are black squares.
The grid is 0-indexed. (0, 0) is bottom-left. ({grid_size - 1}, {grid_size - 1}) is top-right.

**Memory (last 5 moves)**
{history_lines}

**Move Analysis (cell visit frequency)**
{move_analysis}

---

### Question

Should the agent move **{direction}**?

---

### Instructions (read carefully before responding)

1. **Legal actions** - consider if this direction avoids obstacles and is allowed from the current position.
2. **Goal-seeking** - prioritize moving toward the red goal.
3. **Avoid repetition** - if the same move has been repeated without progress, say NO unless it clearly helps.
4. **Trap avoidance** - avoid directions that lead to dead ends or repeated loops.
5. **Output format** - respond with exactly one word: YES or NO. Uppercase, no punctuation, no extra text, not bolded.
   *Do not include any other explanation, characters, or formatting.*

**Important:** If two black obstacle squares touch at the corners (diagonally), a thick black line will connect them.
This indicates that the agent **cannot pass diagonally** between those cells.

You must treat this diagonal as a wall — only cardinal (up/down/left/right) movements are allowed, and diagonal movement is not possible under any condition.

Now respond: YES or NO
"""

def build_yesno_prompt_unassigned_goals(
    agent_id,
    agent_pos,
    goal_positions,
    other_agents,  # list of tuples (other_agent_id, position)
    grid_size,
    obstacles,
    direction,
    memory,   # list of (r0, c0, dir, r1, c1)
    visits    # dict {(r, c): count}
):
    # Format obstacle coordinates
    obs_coords = ', '.join([f"({r}, {c})" for r, c in sorted(obstacles)])

    # Format memory
    if memory:
        history_lines = "\n".join(
            [f"  • {i+1}. you moved from (row {r0}, col {c0}) **{dir_}** → got to (row {r1}, col {c1})"
             for i, (r0, c0, dir_, r1, c1) in enumerate(memory[:5])]
        )
    else:
        history_lines = "  • (no prior moves — this is the first step)"

    # Format move analysis
    move_analysis_lines = []
    for d in ['up', 'down', 'left', 'right']:
        r, c = agent_pos
        if d == "up":
            target = (r + 1, c)
        elif d == "down":
            target = (r - 1, c)
        elif d == "left":
            target = (r, c - 1)
        elif d == "right":
            target = (r, c + 1)
        else:
            continue
        count = visits.get(target, 0)
        move_analysis_lines.append(f"  • {d:5} → (row {target[0]}, col {target[1]}) — visited {count} time(s)")
    move_analysis = "\n".join(move_analysis_lines)

    # Format other agents
    if other_agents:
        other_agent_lines = "\n".join(
            [f"  • Agent {aid} is at (row {pos[0]}, col {pos[1]})" for aid, pos in other_agents]
        )
    else:
        other_agent_lines = "  • (no other agents present)"

    # Format goals (unassigned)
    goal_lines = "\n".join(
        [f"  • Goal {chr(65+i)} is at (row {pos[0]}, col {pos[1]})" for i, pos in enumerate(goal_positions) if pos is not None]
    )

    # Dynamically list only the existing goal labels in the environment description
    existing_goal_labels = [chr(65+i) for i, pos in enumerate(goal_positions) if pos is not None]
    if existing_goal_labels:
        goal_label_str = ", ".join([f"**{label}**" for label in existing_goal_labels])
    else:
        goal_label_str = "(none)"

    return f"""
**Environment**

You are Agent {agent_id} (a blue square labeled **{agent_id}**) on a {grid_size}×{grid_size} grid.  
There are several red squares labeled {goal_label_str}. These are **unassigned goals** — you may approach any of them.  
Black squares are obstacles that **cannot be entered**.  
Other agents may be present — they are also shown as blue squares with numeric labels (1, 2, 3, ...).

Four colored borders define direction:
* green (top) → **up**
* gray (bottom) → **down**
* yellow (left) → **left**
* blue (right) → **right**

All coordinates are zero-indexed:
- (0, 0) is the bottom-left corner
- ({grid_size - 1}, {grid_size - 1}) is the top-right corner

In the image:
- Obstacles are black squares labeled **O**
- Goals are red squares labeled {goal_label_str}
- You are labeled **{agent_id}**
- Other agents are labeled numerically

**Current state**  
* Your position        … **(row {agent_pos[0]}, col {agent_pos[1]})**  
* Obstacles            … {obs_coords or "none"}  
* Other agents         …  
{other_agent_lines}
* Goal locations       …  
{goal_lines}

**Memory (last 5 moves)**  
{history_lines}

**Move Analysis (cell visit frequency)**  
{move_analysis}

---

### Question

Should Agent {agent_id} move **{direction}**?

---

### Instructions (read carefully before responding)

1. **Legal actions** - do not walk into obstacles or off the grid.
2. **Goal coverage** - each goal must be reached by one agent, but **goals are unassigned**.
3. **Coordination assumption** - you cannot communicate with other agents. Avoid chasing the same goal as others if better options exist.
4. **Global objective** - minimize the **total number of steps** for all agents to reach all goals.
5. **Don't be greedy** - choosing the nearest goal isn't always optimal for the team.
6. **Output format** - respond with exactly one word: YES or NO. All caps. No punctuation or extra explanation.
7. **Diagonal wall rule** - if two obstacles touch at corners, a thick black diagonal means you cannot pass through that diagonal.

Now respond: YES or NO
"""

def build_yesno_prompt_unassigned_com(
    agent_id,
    agent_pos,
    goal_positions,
    other_agents,  # list of tuples (other_agent_id, position)
    grid_size,
    obstacles,
    direction,
    memory,   # list of (r0, c0, dir, r1, c1)
    visits,   # dict {(r, c): count}
    agent_targets  # list of target goal labels (e.g., ["A", "B", None])
):
    # Format obstacle coordinates
    obs_coords = ', '.join([f"({r}, {c})" for r, c in sorted(obstacles)])

    # Format memory
    if memory:
        history_lines = "\n".join(
            [f"  • {i+1}. you moved from (row {r0}, col {c0}) **{dir_}** → got to (row {r1}, col {c1})"
             for i, (r0, c0, dir_, r1, c1) in enumerate(memory[:5])]
        )
    else:
        history_lines = "  • (no prior moves — this is the first step)"

    # Format move analysis
    move_analysis_lines = []
    for d in ['up', 'down', 'left', 'right']:
        r, c = agent_pos
        if d == "up":
            target = (r + 1, c)
        elif d == "down":
            target = (r - 1, c)
        elif d == "left":
            target = (r, c - 1)
        elif d == "right":
            target = (r, c + 1)
        else:
            continue
        count = visits.get(target, 0)
        move_analysis_lines.append(f"  • {d:5} → (row {target[0]}, col {target[1]}) — visited {count} time(s)")
    move_analysis = "\n".join(move_analysis_lines)

    # Format other agents
    if other_agents:
        other_agent_lines = "\n".join(
            [f"  • Agent {aid} is at (row {pos[0]}, col {pos[1]})" for aid, pos in other_agents]
        )
    else:
        other_agent_lines = "  • (no other agents present)"

    # Format declared targets
    declared_target_lines = []
    for idx, (aid, tgt) in enumerate(zip(range(1, len(agent_targets) + 1), agent_targets)):
        if aid == agent_id:
            continue  # don't include self
        if tgt is not None and other_agents and any(oaid == aid for oaid, _ in other_agents):
            declared_target_lines.append(f"  • Agent {aid} → Goal {tgt}")
    if declared_target_lines:
        declared_targets_block = "\n".join(declared_target_lines)
    else:
        declared_targets_block = "  • (no goal commitments from other agents)"

    # Format goals (unassigned)
    goal_lines = "\n".join(
        [f"  • Goal {chr(65+i)} is at (row {pos[0]}, col {pos[1]})" for i, pos in enumerate(goal_positions) if pos is not None]
    )

    # Dynamically list only the existing goal labels in the environment description
    existing_goal_labels = [chr(65+i) for i, pos in enumerate(goal_positions) if pos is not None]
    if existing_goal_labels:
        goal_label_str = ", ".join([f"**{label}**" for label in existing_goal_labels])
    else:
        goal_label_str = "(none)"

    return f"""
**Environment**

You are Agent {agent_id} (a blue square labeled **{agent_id}**) on a {grid_size}×{grid_size} grid.  
There are several red squares labeled {goal_label_str}. These are **unassigned goals** — you may approach any of them.  
Black squares are obstacles that **cannot be entered**.  
Other agents may be present — they are also shown as blue squares with numeric labels (1, 2, 3, ...).

Four colored borders define direction:
* green (top) → **up**
* gray (bottom) → **down**
* yellow (left) → **left**
* blue (right) → **right**

All coordinates are zero-indexed:
- (0, 0) is the bottom-left corner
- ({grid_size - 1}, {grid_size - 1}) is the top-right corner

In the image:
- Obstacles are black squares labeled **O**
- Goals are red squares labeled {goal_label_str}
- You are labeled **{agent_id}**
- Other agents are labeled numerically

**Current state**  
* Your position        … **(row {agent_pos[0]}, col {agent_pos[1]})**  
* Obstacles            … {obs_coords or "none"}  
* Other agents         …  
{other_agent_lines}
* Declared targets     …  
{declared_targets_block}
* Goal locations       …  
{goal_lines}

**Memory (last 5 moves)**  
{history_lines}

**Move Analysis (cell visit frequency)**  
{move_analysis}

---

### Question

Should Agent {agent_id} move **{direction}**?
What goal should Agent {agent_id} pursue?

---

### Instructions (read carefully before responding)

1. **Legal actions** - do not walk into obstacles or off the grid.
2. **Goal coverage** - each goal must be reached by one agent, but **goals are unassigned**.
3. **Coordination assumption** - you cannot communicate with other agents. Avoid chasing the same goal as others if better options exist.
4. **Global objective** - minimize the **total number of steps** for all agents to reach all goals.
5. **Don't be greedy** - choosing the nearest goal isn't always optimal for the team.
6. **Output format** - respond with exactly one word: YES or NO. All caps. No punctuation or extra explanation.
7. **Diagonal wall rule** - if two obstacles touch at corners, a thick black diagonal means you cannot pass through that diagonal.
8. **Coordination via targets** - You are aware of other agents' chosen goals. If your selected target goal conflicts with theirs, consider whether **you** should change. Do not change without a reason — prefer to stay on your current goal unless a conflict clearly requires resolution.
9. **Explanation** - Give a brief 1-2 sentence explanation of your reasoning for the move and goal choice in the explanation field.

Now respond: YES or NO
"""

def build_yesno_prompt_unassigned_com_unstructured(
    agent_id,
    agent_pos,
    goal_positions,
    other_agents,  # list of tuples (other_agent_id, position)
    grid_size,
    obstacles,
    direction,
    memory,   # list of (r0, c0, dir, r1, c1)
    visits,   # dict {(r, c): count}
    agent_targets  # list of target goal labels (e.g., ["A", "B", None])
):
    # Format obstacle coordinates
    obs_coords = ', '.join([f"({r}, {c})" for r, c in sorted(obstacles)])

    # Format memory
    if memory:
        history_lines = "\n".join(
            [f"  • {i+1}. you moved from (row {r0}, col {c0}) **{dir_}** → got to (row {r1}, col {c1})"
             for i, (r0, c0, dir_, r1, c1) in enumerate(memory[:5])]
        )
    else:
        history_lines = "  • (no prior moves — this is the first step)"

    # Format move analysis
    move_analysis_lines = []
    for d in ['up', 'down', 'left', 'right']:
        r, c = agent_pos
        if d == "up":
            target = (r + 1, c)
        elif d == "down":
            target = (r - 1, c)
        elif d == "left":
            target = (r, c - 1)
        elif d == "right":
            target = (r, c + 1)
        else:
            continue
        count = visits.get(target, 0)
        move_analysis_lines.append(f"  • {d:5} → (row {target[0]}, col {target[1]}) — visited {count} time(s)")
    move_analysis = "\n".join(move_analysis_lines)

    # Format other agents
    if other_agents:
        other_agent_lines = "\n".join(
            [f"  • Agent {aid} is at (row {pos[0]}, col {pos[1]})" for aid, pos in other_agents]
        )
    else:
        other_agent_lines = "  • (no other agents present)"

    # Format declared targets
    declared_target_lines = []
    for idx, (aid, tgt) in enumerate(zip(range(1, len(agent_targets) + 1), agent_targets)):
        if aid == agent_id:
            continue  # don't include self
        if tgt is not None and other_agents and any(oaid == aid for oaid, _ in other_agents):
            declared_target_lines.append(f"  • Agent {aid} → Goal {tgt}")
    if declared_target_lines:
        declared_targets_block = "\n".join(declared_target_lines)
    else:
        declared_targets_block = "  • (no goal commitments from other agents)"

    # Format goals (unassigned)
    goal_lines = "\n".join(
        [f"  • Goal {chr(65+i)} is at (row {pos[0]}, col {pos[1]})" for i, pos in enumerate(goal_positions) if pos is not None]
    )

    # Dynamically list only the existing goal labels in the environment description
    existing_goal_labels = [chr(65+i) for i, pos in enumerate(goal_positions) if pos is not None]
    if existing_goal_labels:
        goal_label_str = ", ".join([f"**{label}**" for label in existing_goal_labels])
    else:
        goal_label_str = "(none)"

    return f"""
**Environment**

You are Agent {agent_id} (a blue square labeled **{agent_id}**) on a {grid_size}×{grid_size} grid.  
There are several red squares labeled {goal_label_str}. These are **unassigned goals** — you may approach any of them.  
Black squares are obstacles that **cannot be entered**.  
Other agents may be present — they are also shown as blue squares with numeric labels (1, 2, 3, ...).

Four colored borders define direction:
* green (top) → **up**
* gray (bottom) → **down**
* yellow (left) → **left**
* blue (right) → **right**

All coordinates are zero-indexed:
- (0, 0) is the bottom-left corner
- ({grid_size - 1}, {grid_size - 1}) is the top-right corner

In the image:
- Obstacles are black squares labeled **O**
- Goals are red squares labeled {goal_label_str}
- You are labeled **{agent_id}**
- Other agents are labeled numerically

**Current state**  
* Your position        … **(row {agent_pos[0]}, col {agent_pos[1]})**  
* Obstacles            … {obs_coords or "none"}  
* Other agents         …  
{other_agent_lines}
* Declared targets     …  
{declared_targets_block}
* Goal locations       …  
{goal_lines}

**Memory (last 5 moves)**  
{history_lines}

**Move Analysis (cell visit frequency)**  
{move_analysis}

---

### Question

Should Agent {agent_id} move **{direction}**?
What goal should Agent {agent_id} pursue?

---

### Instructions (read carefully before responding)

1. **Legal actions** - do not walk into obstacles or off the grid.
2. **Goal coverage** - each goal must be reached by one agent, but **goals are unassigned**.
3. **Coordination assumption** - you cannot communicate with other agents. Avoid chasing the same goal as others if better options exist.
4. **Global objective** - minimize the **total number of steps** for all agents to reach all goals.
5. **Don't be greedy** - choosing the nearest goal isn't always optimal for the team.
6. **Output format** - respond with exactly one word: YES or NO. All caps. No punctuation or extra explanation.
7. **Diagonal wall rule** - if two obstacles touch at corners, a thick black diagonal means you cannot pass through that diagonal.
8. **Coordination via targets** - You are aware of other agents' chosen goals. If your selected target goal conflicts with theirs, consider whether **you** should change. Do not change without a reason — prefer to stay on your current goal unless a conflict clearly requires resolution.
9. **Explanation** - Give a brief 1-2 sentence explanation of your reasoning for the move and goal choice in the explanation field.
10. **Response structure** - Provide your answer in a JSON format with keys "move", "target", and "explanation". However, do not include the json wrapper with the triple quotes in your response, just the JSON object itself in plain text format. DO NOT WRITE THE TRIPLE QUOTES JSON.

Respond in the JSON format:
{{
  "move": "YES or NO in CAPS",
  "target": "(goal letter, e.g. A, B, ... in CAPS)",
  "explanation": "(brief justification for your choice, 1-2 sentences)"
}}
"""

def build_yesno_prompt_unstruc_v2(
    agent_id,
    agent_pos,
    goal_positions,
    other_agents,  # list of tuples (other_agent_id, position)
    grid_size,
    obstacles,
    direction,
    memory,   # list of (r0, c0, dir, r1, c1)
    visits,   # dict {(r, c): count}
    agent_targets,  # list of target goal labels (e.g., ["A", "B", None])
    env  # full GridWorld environment instance (required for pathfinding)
):
    # Format obstacle coordinates
    obs_coords = ', '.join([f"({r}, {c})" for r, c in sorted(obstacles)])

    # Format memory
    if memory:
        history_lines = "\n".join(
            [f"  • {i+1}. you moved from (row {r0}, col {c0}) **{dir_}** → got to (row {r1}, col {c1})"
             for i, (r0, c0, dir_, r1, c1) in enumerate(memory[:5])]
        )
    else:
        history_lines = "  • (no prior moves — this is the first step)"

    # Format move analysis
    move_analysis_lines = []
    for d in ['up', 'down', 'left', 'right']:
        r, c = agent_pos
        if d == "up":
            target = (r + 1, c)
        elif d == "down":
            target = (r - 1, c)
        elif d == "left":
            target = (r, c - 1)
        elif d == "right":
            target = (r, c + 1)
        else:
            continue
        count = visits.get(target, 0)
        move_analysis_lines.append(f"  • {d:5} → (row {target[0]}, col {target[1]}) — visited {count} time(s)")
    move_analysis = "\n".join(move_analysis_lines)

    # Format other agents
    if other_agents:
        other_agent_lines = "\n".join(
            [f"  • Agent {aid} is at (row {pos[0]}, col {pos[1]})" for aid, pos in other_agents]
        )
    else:
        other_agent_lines = "  • (no other agents present)"

    # Format declared targets
    declared_target_lines = []
    for idx, (aid, tgt) in enumerate(zip(range(1, len(agent_targets) + 1), agent_targets)):
        if aid == agent_id:
            continue  # don't include self
        if tgt is not None and other_agents and any(oaid == aid for oaid, _ in other_agents):
            declared_target_lines.append(f"  • Agent {aid} → Goal {tgt}")
    if declared_target_lines:
        declared_targets_block = "\n".join(declared_target_lines)
    else:
        declared_targets_block = "  • (no goal commitments from other agents)"

    # Format goals (unassigned)
    goal_lines = "\n".join(
        [f"  • Goal {chr(65+i)} is at (row {pos[0]}, col {pos[1]})" for i, pos in enumerate(goal_positions) if pos is not None]
    )

    # Dynamically list only the existing goal labels in the environment description
    existing_goal_labels = [chr(65+i) for i, pos in enumerate(goal_positions) if pos is not None]
    if existing_goal_labels:
        goal_label_str = ", ".join([f"**{label}**" for label in existing_goal_labels])
    else:
        goal_label_str = "(none)"

    # Calculate projected cell number if the agent moves in the proposed direction
    move_offsets = {
        'up': (1, 0),
        'down': (-1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }
    if direction in move_offsets:
        r, c = agent_pos
        dr, dc = move_offsets[direction]
        target_pos = (r + dr, c + dc)
        if 0 <= target_pos[0] < grid_size and 0 <= target_pos[1] < grid_size:
            target_cell_id = target_pos[0] * grid_size + target_pos[1]
            move_label_line = f"If you move **{direction}**, you will arrive at cell **{target_cell_id}**."
        else:
            move_label_line = f"If you move **{direction}**, you would go out of bounds."
    else:
        move_label_line = ""

    return f"""
**Environment**

You are Agent {agent_id} (a blue **circle** labeled **{agent_id}**) on a {grid_size}×{grid_size} grid.  
There are several red squares labeled {goal_label_str}. These are **unassigned goals** — you may approach any of them.  
Black squares are obstacles that **cannot be entered**.  
Other agents may be present — they are also shown as blue circles with numeric labels (1, 2, 3, ...).  
All empty cells are labeled with a gray number in the background to assist reasoning.  
Cell labels increase from the **bottom-left**, moving **left to right**, then **up row by row**.

Four colored borders define direction:
* green (top) → **up**
* gray (bottom) → **down**
* yellow (left) → **left**
* blue (right) → **right**

All coordinates are zero-indexed:
- (0, 0) is the bottom-left corner
- ({grid_size - 1}, {grid_size - 1}) is the top-right corner

In the image:
- Obstacles are black squares labeled **O**
- Goals are red squares labeled {goal_label_str}
- You are labeled **{agent_id}**
- Other agents are labeled numerically

A simulation step means that every agent takes one move at the same time. The simulation ends when all agents have reached a goal. Your job is to minimize the total number of simulation steps — the number of moves until the last agent finishes.
Your goal is to minimize the total number of simulation steps required for all agents to reach the goals.

A simulation step means that all agents move once at the same time. The simulation ends when every agent has reached a goal. Therefore, the total number of steps is determined by the agent that takes the most steps to reach their goal. In other words:

    Total steps = the number of steps taken by the last agent to reach a goal.

❗ This has important consequences:

    You should not just minimize your own distance to a goal.

    Sometimes, it is better to go to a further goal so that another agent can take a closer goal, which reduces how long they need to move.

    If you take the closest goal selfishly, another agent might be forced to take a distant one, increasing the total simulation time.

    Your decision should consider the positions of all agents and goals, and aim for the best global assignment.

✅ Example:

    You are 3 steps from Goal A and 6 steps from Goal B.

    Agent 2 is 6 steps from Goal A and 4 steps from Goal B.

    If you take Goal A and Agent 2 takes Goal B, the simulation ends in 4 steps (you finish in 3, Agent 2 in 4).

    If you take Goal B and Agent 2 takes Goal A, the simulation ends in 6 steps.

    So you should take Goal A — not because it's closer, but because this choice leads to the shortest total simulation.

Your reasoning and goal selection should be based on this principle.

**Current state**  
* Your position        … **(row {agent_pos[0]}, col {agent_pos[1]})**  
* Obstacles            … {obs_coords or "none"}  
* Other agents         …  
{other_agent_lines}
* Declared targets     …  
{declared_targets_block}
* Goal locations       …  
{goal_lines}

**Memory (last 5 moves)**  
{history_lines}

**Move Analysis (cell visit frequency)**  
{move_analysis}

---

### Question

Should Agent {agent_id} move **{direction}**?  
{move_label_line}  
What goal should Agent {agent_id} pursue?

---

### Instructions (read carefully before responding)

1. **Legal actions** - do not walk into obstacles or off the grid.
2. **Goal coverage** - each goal must be reached by one agent, but **goals are unassigned**.
3. **Coordination assumption** - you cannot communicate with other agents. Avoid chasing the same goal as others if better options exist.
4. **Team objective** - The goal is to minimize the total number of simulation steps. In this simulation, each step means all agents move once, and the total is the number of steps until all agents have reached their goals. So if one agent takes 2 steps and another takes 8, the total is 8 - the time it takes for the last agent to finish.
5. **Don't be greedy** - DO NOT just go to the closest goal. Think about the whole team. Sometimes it is better for you to take a longer path so that another agent can reach a closer goal faster. Your goal is to minimize the number of total simulation steps, not just your own effort. Thus, you are minimizing the time it takes for the **last/slowest** agent to finish.
6. **Output format** - respond with exactly one word: YES or NO. All caps. No punctuation or extra explanation.
7. **Diagonal wall rule** - if two obstacles touch at corners, a thick black diagonal means you cannot pass through that diagonal.
8. **Coordination via targets** - You are aware of other agents' chosen goals. If your selected target goal conflicts with theirs, consider whether **you** should change. Do not change without a reason — prefer to stay on your current goal unless a conflict clearly requires resolution.
9. **Explanation** - Give a brief 1-2 sentence explanation of your reasoning for the move and goal choice in the explanation field.
10. **Response structure** - Provide your answer in a JSON format with keys "move", "target", and "explanation". However, do not include the json wrapper with the triple quotes in your response, just the JSON object itself in plain text format. DO NOT WRITE THE TRIPLE QUOTES JSON.
11. Example - Suppose there are 2 goals:

    Goal A is 3 steps from you, 6 steps from Agent 2

    Goal B is 5 steps from you, 4 steps from Agent 2

If you take Goal A, Agent 2 must go to Goal B, which takes 4 steps. You both finish in 5 steps.
But if you go to Goal B, and Agent 2 takes Goal A, you finish in 5, and Agent 2 takes 6. The total is 6 — higher.

The total number of steps = the number of steps it takes for the slowest agent to finish.
You must reason about who should take which goal to minimize the total number of simulation steps.
Try to also think about the **long-term** consequences of your move, not just the immediate distance. If you take a goal, how does it affect the other agents' choices? Will it lead to a longer total time for the team?
Do not pick the furthest goal just to avoid conflict. Instead, think about the overall team efficiency, since if you pick a goal that's further for you than another agent, it might lead to a longer total time for the team because of YOU. They should pick that goal instead. You should pick the goal that might be closer to you and they should pick that's further for them, if it leads to a shorter total time for the team.
Don't be too much of a pushover, but also don't be too selfish. Think about the team and how to minimize the total number of simulation steps.

DO NOT AVOID CONFLICTS JUST TO AVOID THEM.
    If you see a conflict, think about whether it is better for you to change your goal or for the other agent to change theirs. If you can resolve the conflict in a way that minimizes the total number of simulation steps, do so.

Respond in the JSON format:
{{
  "move": "YES or NO in CAPS",
  "target": "(goal letter, e.g. A, B, ... in CAPS)",
  "explanation": "(brief justification for your choice, 1-2 sentences)"
}}
"""

def build_target_selection_prompt(
    agent_id,
    agent_pos,
    goal_positions,
    other_agents,
    grid_size,
    obstacles,
    memory,
    visits,
    agent_targets,
    target_memory=None,
):
    obs_coords = ', '.join(f"({r}, {c})" for r, c in sorted(obstacles))

    if memory:
        history_lines = "\n".join(
            f"  • {i+1}. You moved from (row {r0}, col {c0}) **{d}** → (row {r1}, col {c1})"
            for i, (r0, c0, d, r1, c1) in enumerate(memory[:5])
        )
    else:
        history_lines = "  • (no prior moves — this is the first step)"

    goal_lines = "\n".join(
        f"  • Goal {chr(65+i)} is at (row {r}, col {c})"
        for i, pos in enumerate(goal_positions)
        if pos is not None
        for r, c in [pos]
    )

    other_agent_lines = (
        "\n".join(
            f"  • Agent {aid} is at (row {p[0]}, col {p[1]})"
            for aid, p in other_agents
        )
        if other_agents
        else "  • (no other agents present)"
    )

    declared_target_lines = [
        f"  • Agent {aid} → Goal {tgt}"
        for aid, tgt in zip(range(1, len(agent_targets) + 1), agent_targets)
        if aid != agent_id and tgt and any(aid == oa[0] for oa in other_agents)
    ]
    declared_targets_block = (
        "\n".join(declared_target_lines)
        if declared_target_lines
        else "  • (no goal commitments from other agents)"
    )

    if target_memory:
        past_targets_lines = "\n".join(
            f"  • Step {step}: chose Goal {tgt} — {ex}"
            for step, tgt, ex in target_memory[-5:]
        )
    else:
        past_targets_lines = "  • (no prior target selections)"

    return f"""
**Environment**

You are Agent {agent_id} (blue circle **{agent_id}**) on a {grid_size}×{grid_size} grid.  
Choose **one** goal to pursue (red squares A, B, C…).  
Obstacles: black squares. Empty cells: gray numbers (left→right, bottom→top).

**Simulation mechanics**

- All agents move **simultaneously** each timestep.  
- The run ends when **every** goal is reached.  
- **Total cost = number of timesteps until the *last* agent finishes.**

Your objective is to **minimise this total cost**, not merely your own distance.

---

**Grid layout**  
• The world is a square {grid_size} × {grid_size} grid.  
• **Coordinates are zero-indexed**: (row 0, col 0) is the *bottom-left*; (row {grid_size-1}, col {grid_size-1}) is the *top-right*.  
• Each empty cell shows its coordinate in light-grey text – formatted “row,col” (e.g. `02,05` means row 2, col 5).  
• Horizontal rows are numbered upward; vertical columns numbered left→right.  

**Cell types & colours**  
| Colour / glyph | Meaning | Example label | Notes |  
|----------------|---------|---------------|-------|  
| 🔵 Blue circle with white number | *Agent* (that number is their ID) | 1 | You are one of these. |  
| 🟥 Solid red square | *Unassigned goal* | A, B, C… | Any agent may pursue; each goal must be claimed by exactly one agent. |  
| ⬛ Black square (“O”) | *Obstacle* | O | Impassable. Cannot stand on or move through. |  
| ◻️ Light-grey “row,col” | Empty cell | `04,07` | Traversable. |  

**Coloured border clues**  
• Top border = green (↑ up) • Bottom = orange (↓ down)  
• Left = yellow (← left)   • Right = blue (→ right)  

**Diagonal wall rule**  
If two black squares touch only at a corner, a thick black diagonal bar is drawn – you cannot cut through that diagonal.  

**Turn-based movement**  
• Time advances in **timesteps**.  
• In each timestep **all agents move simultaneously** (or stay).  
• Valid moves: up, down, left, right (one cell). You cannot enter obstacles or off-grid cells.  

**When does the simulation end?**  
– As soon as *every* goal has an agent standing on it.  

**Team cost metric**  
– We measure the total runtime as the **number of timesteps until the last agent reaches its goal**.  
– Your objective is to choose goals (and later moves) so that this “last-agent” finish time is as small as possible; minimise the **maximum** path length among all agents.  

**Why goal choice matters**  
– Picking the closest goal for yourself is sometimes *worse* for the team, because it may force another agent onto a very long route.  
– Always compare alternative assignments and pick the one with the **shorter longest path** – even if that means taking a slightly farther goal personally.

---

### ➊ Key reasoning rules

1. **No duplication**: each goal should end up with exactly one agent.  
2. **Greedy ≠ optimal**: sometimes you must pick a farther goal so another agent can finish sooner.  
3. **Don’t be a pushover**: if you’re clearly the best-placed agent for a goal, keep it unless switching lowers total cost.  
4. Use relative distances, obstacles, and potential path conflicts to decide.
5. Try to think of different assignments and their longest paths and then evaluate which one is the best for you and the team, do not get stuck on one assignment and its explanation.

---

### ➋ Team-level reasoning checklist ✅  
(Think silently through these steps before answering.)

1. List every **remaining goal** and estimate which agent is currently fastest to reach it.  
2. Draft a **full assignment** (agents → goals, no duplicates).  
3. Compute that assignment’s **longest path length** (this defines total cost).  
4. Try at least one alternative assignment; see if the longest path shrinks.  
5. Select the goal for **you** that belongs to the assignment with the **smallest longest-path**.  
6. Apply conflict guidelines (below) to decide whether to keep or switch when ties/conflicts arise.

---

### ➌ Conflict-resolution guidelines 🚦  
*(Two or more agents want the same goal — what now?)*  

1. **Enumerate both assignments**  
   *A1 → G₁ & A2 → G₂* **vs.** *A1 → G₂ & A2 → G₁* (swap the contested goal).  

2. **Compute each agent’s path length** for both assignments.  

3. **Find the longest path** in each assignment (that’s the team’s total time).  

4. **Pick the assignment with the smaller longest-path**, even if that means **you** take the farther goal.  

#### Worked conflict example ⚖️  

```

Grid (rows shown top-down)        Key
1 . . . . . 2 . . .               1,2  = agents
. . # . . . B . # A               A,B  = goals
. . . . . . . . .                 #    = obstacle

````

*Distances (in steps avoiding obstacles)*  

| Agent | → Goal A | → Goal B |
|-------|----------|----------|
| **1** | 10       | 7        |
| **2** | 4        | 1        |

**Team-optimal choice** (Agent 2 takes nearest)  
*1 → A (10)*, *2 → B (1)* ⇒ longest path = **10**  ❌ worse  

**Greedy choice** (Agent 1 takes nearest, Agent 2 let's it happen even though it's only 1 step away)  
*1 → B (7)*, *2 → A (4)* ⇒ longest path = **7**  

But reverse the distances and obstacles and the result can flip.  
**Always run the four-step checklist above; choose the assignment with the smaller longest path — even if that means taking the farther goal yourself.**

---

### ➍ Worked examples

**Example A — keep your goal**  
- You → Goal B: 3 steps  
- Agent 2 → Goal B: 5 steps  
Alternative assignments give total ≥ 5 steps.  
👉 You keep Goal B; Agent 2 switches.

**Example B — switch goals**  
- You → Goal C: 8 steps  
- Agent 3 → Goal C: 4 steps  
Switching: You → Goal A (6 steps) & Agent 3 stays on C → longest path = 6 < 8.  
👉 You switch to Goal A.

---

**Current state**  
* Your position … **(row {agent_pos[0]}, col {agent_pos[1]})**  
* Obstacles … {obs_coords or "none"}  
* Other agents …  
{other_agent_lines}
* Declared targets …  
{declared_targets_block}
* Goal locations …  
{goal_lines}

**Memory (last 5 moves)**  
{history_lines}

**Previous target selections**  
{past_targets_lines}

---

### Question

Which goal should Agent {agent_id} pursue **now**?  
Return **only** JSON:

```json
{{
  "reasoning": "Step-by-step thoughts: consider agent distances, conflicts, tradeoffs, and explain your decision path.",
  "explanation": "One or two sentence summary of your final goal choice.",
  "target": "Goal letter (e.g., A, B, C)"
}}
````

"""


def build_direction_selection_prompt(
    agent_id,
    agent_pos,
    declared_goal,  # e.g., "B"
    goal_positions,  # list of (r, c)
    other_agents,  # list of (id, (r, c))
    grid_size,
    obstacles,
    direction,  # proposed direction
    memory,
    visits,
    agent_targets  # list of goal letters for all agents
):
    obs_coords = ', '.join([f"({r}, {c})" for r, c in sorted(obstacles)])

    # Move history
    if memory:
        history_lines = "\n".join(
            [f"  • {i+1}. you moved from (row {r0}, col {c0}) **{dir_}** → (row {r1}, col {c1})"
             for i, (r0, c0, dir_, r1, c1) in enumerate(memory[:5])]
        )
    else:
        history_lines = "  • (no prior moves — this is the first step)"

    # Move analysis
    move_analysis_lines = []
    for d in ['up', 'down', 'left', 'right']:
        r, c = agent_pos
        if d == "up":
            target = (r + 1, c)
        elif d == "down":
            target = (r - 1, c)
        elif d == "left":
            target = (r, c - 1)
        elif d == "right":
            target = (r, c + 1)
        else:
            continue
        count = visits.get(target, 0)
        move_analysis_lines.append(f"  • {d:5} → (row {target[0]}, col {target[1]}) — visited {count} time(s)")
    move_analysis = "\n".join(move_analysis_lines)

    # Calculate projected cell number
    move_offsets = {
        'up': (1, 0),
        'down': (-1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }
    move_label_line = ""
    if direction in move_offsets:
        r, c = agent_pos
        dr, dc = move_offsets[direction]
        target_pos = (r + dr, c + dc)
        if 0 <= target_pos[0] < grid_size and 0 <= target_pos[1] < grid_size:
            cell_id = target_pos[0] * grid_size + target_pos[1]
            move_label_line = f"If you move **{direction}**, you will arrive at cell **{cell_id}**."
        else:
            move_label_line = f"If you move **{direction}**, you would go out of bounds."

    # Declared goal location
    if declared_goal is not None:
        goal_index = ord(declared_goal.upper()) - 65
        goal_pos = goal_positions[goal_index] if 0 <= goal_index < len(goal_positions) else None
        if goal_pos is not None:
            goal_line = (
                f"* Your current target goal is **Goal {declared_goal}**, located at (row {goal_pos[0]}, col {goal_pos[1]}). "
                "Only you should enter this goal. Do not enter goals assigned to other agents."
            )
        else:
            goal_line = "* You have a declared goal, but its location is not known."
    else:
        goal_line = "* You have not declared a goal yet."


    # Declared targets of others
    declared_targets_block = []
    for aid, tgt in zip(range(1, len(agent_targets)+1), agent_targets):
        if aid == agent_id:
            continue
        if tgt is not None and any(oaid == aid for oaid, _ in other_agents):
            declared_targets_block.append(f"  • Agent {aid} → Goal {tgt}")
    declared_block = "\n".join(declared_targets_block) if declared_targets_block else "  • (no goal commitments from other agents)"

    return f"""
**Environment**

You are Agent {agent_id} (a blue circle labeled **{agent_id}**) on a {grid_size}×{grid_size} grid.  
Your current declared target is Goal {declared_goal}.  
Your job now is to decide whether you should move **{direction}** to approach it.  

Other agents are also choosing their moves. Coordination is important — do not walk into obstacles or other agents.  

All empty cells are labeled with a gray number for orientation.  
Cell numbers increase left to right, then row by row from bottom to top.  

**Current state**  
* Your position        … **(row {agent_pos[0]}, col {agent_pos[1]})**  
* {goal_line}  
* Obstacles            … {obs_coords or "none"}  
* Other agents         …  
{chr(10).join([f"  • Agent {aid} is at (row {r}, col {c})" for aid, (r, c) in other_agents]) or "  • (none)"}  
* Declared targets     …  
{declared_block}

* Goal assignments … Each goal must be reached by exactly one agent. Avoid stepping into another agent’s goal.

**Memory (last 5 moves)**  
{history_lines}

**Move Analysis (visit frequency)**  
{move_analysis}

---

### Question

Should Agent {agent_id} move **{direction}**?  
{move_label_line}

---

### Instructions

1. Only respond YES if the move brings you closer to your own goal and avoids danger.
2. DO NOT move into obstacles or off the grid.
3. DO NOT enter a goal cell that is assigned to another agent, even if it's closer.
4. If another agent is nearby or heading to the same area, consider avoiding a collision.
5. Use your declared goal to reason about where to go — you are committed to it unless a new plan is made.
6. Do not block others if there’s a better route for the team.

Respond in the JSON format:
```json
{{
  "move": "YES or NO in CAPS",
  "explanation": "(brief justification for your choice, 1-2 sentences)"
}}
```
"""