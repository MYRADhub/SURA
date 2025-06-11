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
