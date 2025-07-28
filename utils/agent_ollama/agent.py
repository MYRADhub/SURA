import time
import json
import csv
from environment import GridWorld
from plot import plot_grid_unassigned_labeled
from collections import deque
import re
import argparse
import random
import ollama
import base64
from pydantic import BaseModel
from typing import List

MODEL = 'llava'

class RankingResponse(BaseModel):
    reasoning: str
    explanation: str
    ranking: List[str]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def send_image_to_model_ollama(image_path, prompt, model='llava', schema=None):
    kwargs = {
        'model': model,
        'messages': [
            {
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }
        ]
    }
    if schema:
        kwargs['format'] = schema

    response = ollama.chat(**kwargs)
    return prompt, response['message']['content'].strip()


def shortest_path_length(start, goal, env):
    if start == goal:
        return 0
    visited = set()
    queue = deque([(start, 0)])
    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == goal:
            return dist
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            next_pos = (nr, nc)
            if (
                0 <= nr < env.size and 0 <= nc < env.size and
                next_pos not in visited and
                next_pos not in env.obstacles
            ):
                visited.add(next_pos)
                queue.append((next_pos, dist + 1))
    return float('inf')  # No path found


def select_direction_opt(agent_pos, declared_goal, goal_positions, env):
    """
    Select the direction that reduces the distance to the target goal the fastest.
    """
    if not declared_goal:
        return None

    goal_index = ord(declared_goal.upper()) - 65
    if goal_index >= len(goal_positions) or goal_positions[goal_index] is None:
        return None

    target_goal = goal_positions[goal_index]
    best_dir = None
    best_dist = float('inf')

    directions = {
        "up": (1, 0),
        "down": (-1, 0),
        "left": (0, -1),
        "right": (0, 1)
    }

    for dir_str, (dr, dc) in directions.items():
        new_pos = (agent_pos[0] + dr, agent_pos[1] + dc)
        if env.is_valid(new_pos):
            dist = shortest_path_length(new_pos, target_goal, env)
            if dist < best_dist:
                best_dist = dist
                best_dir = dir_str
    return best_dir

def build_target_ranking_prompt(
    agent_id,
    agent_pos,
    goal_positions,
    other_agents,
    grid_size,
    obstacles,
    memory,
    visits,
    agent_targets,
    target_memory,
    distances
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

    distance_table_lines = []
    for aid, dist_list in distances.items():
        row = f"  • Agent {aid}: " + ", ".join(
            f"{chr(65 + i)} = {d if d != float('inf') else '∞'}"
            for i, d in enumerate(dist_list)
        )
        distance_table_lines.append(row)
    distance_block = "\n".join(distance_table_lines)

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
• Each empty cell shows its index in light-grey text that is a number (e.g. `012` means cell 12).  
• The numbers increase from left to right and from bottom to top.  

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
5. Try to think of different assignments and their longest paths and then evaluate which one is the best for you and the team, do not get stuck on one assignment an its explanation.
6. Carefully calculate the full DETAILED step-by-step path length for each agent-goal pair, including all detours around obstacles. Calculate cell-by-cell path analysis using the visual map. RELY ON THE IMAGE MORE.

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

### Goal Ranking Guidelines 🧠

Instead of choosing a single goal, **rank the remaining goals** in order of how well they contribute to team performance.  
This ranking will be used later to assign goals so that **each agent gets one unique goal** and total steps are minimized.

Your ranking should consider:
- How quickly each agent can reach each goal.
- Which assignments minimize the **maximum** path.
- Reasonable fallback options in case of conflict.

Even if you are only 1 step away from a goal, that doesn't necessarily mean you should choose it.
Sometimes it is better to let another agent take a closer goal if it leads to a shorter total
And even if you are further to a goal than another agent, it might be better for you to take that goal if it leads to a shorter total time for the team.
Do not be a pushover, but also do not be too selfish. Think about the team and how to minimize the total number of simulation steps.

The team's total cost is always defined by the agent who finishes last. Always consider alternative assignments: sometimes it's better for an agent to take a slightly farther goal so that another agent avoids a much longer route, even if it means someone gets a non-nearest goal.

For at least two possible ways to assign agents to goals (not just by nearest), list the resulting maximum path length for each assignment. Pick the assignment that minimizes the maximum path length, and only then choose your preferred goal.

For every assignment, check if swapping your choice with another agent's leads to a lower maximum path length.

Rank goals for yourself in order of which assignment leads to the lowest maximum team finish time, not by your own shortest path.
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

**Agent-to-Goal Distances (in steps)**
{distance_block}

---

### Question

Rank the remaining goals for Agent {agent_id} from most to least preferred — based on team-optimal coordination. Return your top-to-bottom preferences.
YOU HAVE TO RANK ALL GOALS, SO IN YOUR RANKING ALL GOALS MUST BE PRESENT, EVEN IF YOU THINK SOME ARE NOT RELEVANT.

Return **only** JSON:
```json
{{
  "reasoning": "Step-by-step thoughts: consider agent distances, conflicts, tradeoffs, and explain your decision path.",
  "explanation": "One or two sentence summary of your final goal choice.",
  "ranking": Ranked list from most to least preferred goal (e.g., ["B", "A", “C”] or [“A”, “D”, “B”, “C”], without any back slashes, without spaces at start and end of the list, without quotes around the list). DO NOT SAY “[B, A, C]” or [ “B”, “A”, “C” ] or [B, A, C], you have to have quotes around the letters for the correct parsing, BUT YOU CAN SAY [“B”, “A”, “C”]
}}
```
"""

def parse_ranking_response(text):
    try:
        parsed = RankingResponse.model_validate_json(text)
        ranking = [g.strip().upper() for g in parsed.ranking if isinstance(g, str)]
        return ranking, parsed.explanation.strip(), parsed.reasoning.strip()
    except Exception as e:
        with open("fails.txt", "a") as f:
            f.write(f"[Ranking Parsing Error] {e}\n{text}\n\n")
        return None, "", ""



def select_target(
    agent_id,
    agent_pos,
    goal_positions,
    other_agents,
    grid_size,
    obstacles,
    memory,
    visits,
    agent_targets,
    target_memory,
    image_path,
    step,
    distances
):
    prompt = build_target_ranking_prompt(
        agent_id=agent_id,
        agent_pos=agent_pos,
        goal_positions=goal_positions,
        other_agents=other_agents,
        grid_size=grid_size,
        obstacles=obstacles,
        memory=memory,
        visits=visits,
        agent_targets=agent_targets,
        target_memory=target_memory,
        distances=distances
    )
    time.sleep(0.5)
    _, response = send_image_to_model_ollama(
        image_path=image_path,
        prompt=prompt,
        model=MODEL,
        schema=RankingResponse.model_json_schema()
    )

    # print(response)

    raw_ranking, explanation, reasoning = parse_ranking_response(response)

    valid_goals = [chr(65 + i) for i, g in enumerate(goal_positions) if g is not None]

    if raw_ranking is None or not raw_ranking:
        # Fallback: random shuffle of valid goals
        fallback_ranking = random.sample(valid_goals, k=len(valid_goals))
        print(f"⚠️ Agent {agent_id} ranking failed. Random fallback: {fallback_ranking}")
        filtered_ranking = fallback_ranking
    else:
        seen = set()
        filtered_ranking = [g for g in raw_ranking if g in valid_goals and not (g in seen or seen.add(g))]

        if not filtered_ranking:
            fallback_ranking = random.sample(valid_goals, k=len(valid_goals))
            print(f"⚠️ Agent {agent_id} gave invalid goal names. Random fallback: {fallback_ranking}")
            filtered_ranking = fallback_ranking


    print(f"Agent {agent_id} ranking: {filtered_ranking}")
    if reasoning:
        print(f"Reasoning: {reasoning}")
    if explanation:
        print(f"Summary: {explanation}")

    if filtered_ranking:
        top = filtered_ranking[0]
        target_memory.append((step, top, explanation))
        if len(target_memory) > 5:
            target_memory[:] = target_memory[-5:]

    return filtered_ranking, explanation, reasoning

def resolve_conflicts(agent_rankings, active_agents):
    final_goals = [rank[0] if rank else None for rank in agent_rankings]
    positions = [0 for _ in agent_rankings]

    while True:
        goal_to_agents = {}
        for idx, goal in enumerate(final_goals):
            if active_agents[idx] and goal:
                goal_to_agents.setdefault(goal, []).append(idx)

        conflicts_exist = any(len(lst) > 1 for lst in goal_to_agents.values())
        if not conflicts_exist:
            break

        for goal, agents in goal_to_agents.items():
            if len(agents) <= 1:
                continue
            agents.sort()  # Ensure lower index agent wins
            for loser_idx in agents[1:]:
                positions[loser_idx] += 1
                if positions[loser_idx] < len(agent_rankings[loser_idx]):
                    final_goals[loser_idx] = agent_rankings[loser_idx][positions[loser_idx]]
                else:
                    final_goals[loser_idx] = None
    return final_goals

def run(
    image_path="grid.png",
    log_path="agent_rank_logs.csv",
    max_steps=100,
    config_path=None,
    obstacles={(2, 2), (3, 3), (4, 1)},
    grid_size=6,
    num_agents=3,
    agent_starts: list[tuple[int, int]] = None,
    goal_positions: list[tuple[int, int]] = None
):
    if config_path:
        env = GridWorld(config_path)
        grid_size = env.size
        num_agents = len(env.agents)
    else:
        env = GridWorld(grid_size, obstacles=obstacles)
        if agent_starts and goal_positions:
            env.initialize_agents_goals_custom(agents=agent_starts, goals=goal_positions)
        else:
            env.initialize_agents_goals(num_agents=num_agents)

    agent_positions = env.agents[:]
    agent_ids = list(range(1, num_agents + 1))
    active = [True] * num_agents
    visits = [{} for _ in range(num_agents)]
    memories = [[] for _ in range(num_agents)]
    target_memories = [[] for _ in range(num_agents)]
    target_goals = [None for _ in range(num_agents)]
    step = 0
    collisions = 0
    log_rows = []
    agent_rankings = [[] for _ in range(num_agents)]

    total_opt = sum(
        min([shortest_path_length(start, g, env) for g in env.goals if g is not None], default=0)
        for start in env.agents if start is not None
    )

    print(f"Initial agent positions: {agent_positions}")
    print(f"Goal positions: {env.goals}")
    print(f"Obstacles: {obstacles}")

    plot_grid_unassigned_labeled(env, image_path=image_path)

    # ----------- Phase 1: Ranking (ONLY ONCE) ----------
    for i in range(num_agents):
        if not active[i] or agent_positions[i] is None:
            continue

        agent_id = agent_ids[i]
        agent_pos = agent_positions[i]
        visits[i][agent_pos] = visits[i].get(agent_pos, 0) + 1

        other_infos = [
            (agent_ids[j], agent_positions[j])
            for j in range(num_agents)
            if j != i and agent_positions[j] is not None
        ]

        distances = {
            agent_ids[j]: [
                shortest_path_length(agent_positions[j], goal, env) if agent_positions[j] and goal else float("inf")
                for goal in env.goals
            ]
            for j in range(num_agents)
        }

        ranking, _, _ = select_target(
            agent_id=agent_id,
            agent_pos=agent_pos,
            goal_positions=env.goals,
            other_agents=other_infos,
            grid_size=grid_size,
            obstacles=obstacles,
            memory=memories[i],
            visits=visits[i],
            agent_targets=target_goals,
            target_memory=target_memories[i],
            image_path=image_path,
            step=step,
            distances=distances
        )
        agent_rankings[i] = ranking

    # ----------- Phase 2: Conflict Resolution (ONLY ONCE) ----------
    proposed_goals = [rank[0] if rank else None for rank in agent_rankings]

    goal_to_agents = {}
    for idx, tgt in enumerate(proposed_goals):
        if active[idx] and tgt:
            goal_to_agents.setdefault(tgt, []).append(idx)

    conflict_pairs = [
        (a1, a2, goal)
        for goal, agents in goal_to_agents.items()
        if len(agents) == 2
        for a1, a2 in [tuple(sorted(agents))]
    ]
    print("Conflict pairs detected:", conflict_pairs)
    proposed_goals = resolve_conflicts(agent_rankings, active)
    print("Proposed goals after conflict resolution:", proposed_goals)


    # Main loop
    while any(active) and step < max_steps:
        print(f"\n--- Step {step} ---")
        plot_grid_unassigned_labeled(env, image_path=image_path)
        proposals = agent_positions[:]

        # ----------- Phase 3: Direction Selection & Movement ----------
        for i in range(num_agents):
            if not active[i] or agent_positions[i] is None:
                continue

            agent_id = agent_ids[i]
            agent_pos = agent_positions[i]
            new_target = proposed_goals[i]

            other_infos = [
                (agent_ids[j], agent_positions[j])
                for j in range(num_agents)
                if j != i and agent_positions[j] is not None
            ]

            best = select_direction_opt(
                agent_pos,
                new_target,
                env.goals,
                env
            )
            explanation = "Chose shortest path step deterministically."

            if best:
                proposals[i] = env.move_agent(agent_pos, best)
                log_rows.append({
                    "step": step,
                    "agent_id": agent_id,
                    "position_before": agent_pos,
                    "position_after": proposals[i],
                    "chosen_direction": best,
                    "explanation": explanation,
                })
                memories[i].append((agent_pos[0], agent_pos[1], best, proposals[i][0], proposals[i][1]))
                if len(memories[i]) > 5:
                    memories[i] = memories[i][-5:]
            else:
                proposals[i] = agent_pos

        target_goals = proposed_goals[:]

        # ----------- Collision Resolution ----------
        new_positions = proposals[:]
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if new_positions[i] == new_positions[j] and agent_positions[i] is not None and agent_positions[j] is not None:
                    collisions += 1
                    new_positions[j] = agent_positions[j]

        # ----------- Face-to-face Swap Conflict Prevention -----------
        # After normal collision resolution, prevent head-on swaps
        move_from = {i: agent_positions[i] for i in range(num_agents)}
        move_to = {i: new_positions[i] for i in range(num_agents)}
        swaps = []
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if (
                    move_to[i] == move_from[j] and
                    move_to[j] == move_from[i] and
                    move_from[i] != move_to[i]  # both actually tried to move
                ):
                    swaps.append((i, j))

        # Force both agents to stay put if they try to swap places
        for i, j in swaps:
            new_positions[i] = agent_positions[i]
            new_positions[j] = agent_positions[j]

        agent_positions = new_positions
        env.agents = agent_positions[:]

        # ----------- Goal Claiming ----------
        to_remove = []
        claimed_goals = []
        for i in range(num_agents):
            if active[i] and agent_positions[i] in env.goals:
                print(f"Agent {agent_ids[i]} reached goal at {agent_positions[i]}")
                active[i] = False
                to_remove.append(i)
                claimed_goals.append(agent_positions[i])

        for i in to_remove:
            agent_positions[i] = None
            env.agents[i] = None
            target_goals[i] = None

        for goal in claimed_goals:
            if goal in env.goals:
                env.goals[env.goals.index(goal)] = None

        print(f"Remaining agents: {[agent_ids[i] for i in range(num_agents) if active[i]]}")
        print(f"Remaining goals: {env.goals}")
        step += 1

    failed = step >= max_steps
    print(f"\nRun completed in {step} steps. Collisions: {collisions}. Failed: {failed}")
    if log_rows:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
    return step, total_opt, failed, collisions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agent rank simulation.")
    parser.add_argument("--config", type=str, default="configs/case_1.yaml", help="Path to config YAML file")
    args = parser.parse_args()

    steps, optimal, failed, collisions = run(config_path=args.config)
    print(f"\n✅ Done!\nOptimal: {optimal}, Steps: {steps}, Failed: {failed}, Collisions: {collisions}")
