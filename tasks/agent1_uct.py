from core.environment import GridWorld
from core.plot import plot_grid
from core.prompt import (
    build_yesno_prompt_single_obs,
    build_prompt_single_obs,
)
from core.agent import send_image_to_model_openai_logprobs
from core.utils import shortest_path_length
from math import sqrt, exp, log


def extract_logprob_multichoice(logprobs, valid_actions):
    """Extracts logprobs for valid actions from top_logprobs (first token only)."""
    logprob_dict = {}
    if not logprobs:
        return {a: float('-inf') for a in valid_actions}

    top_logprobs = logprobs[0].top_logprobs
    for item in top_logprobs:
        tok = item.token
        if tok in valid_actions:
            logprob_dict[tok] = item.logprob

    # Fill in missing actions with -inf
    for a in valid_actions:
        if a not in logprob_dict:
            logprob_dict[a] = float('-inf')
    return logprob_dict


def extract_logprob_yesno(logprobs):
    """Returns logprob for 'YES' token from top_logprobs."""
    if not logprobs:
        return float('-inf')

    for item in logprobs[0].top_logprobs:
        if item.token == "YES":
            return item.logprob
    return float('-inf')


def normalize_logprobs_to_logprobs(logprobs_dict):
    """Converts raw logprobs to normalized logprobs (log softmax)."""
    max_logprob = max(logprobs_dict.values())  # for numerical stability
    exp_probs = {a: exp(lp - max_logprob) for a, lp in logprobs_dict.items()}
    total = sum(exp_probs.values())
    norm_probs = {a: p / total for a, p in exp_probs.items()}
    return {a: log(p) for a, p in norm_probs.items()}  # re-log after softmax


def select_action_ucg(logprobs_dict, visit_counts, state, c=2.0):
    """Selects the action using UCT-style formula based on confidence and visits."""
    scores = {}
    total_visits = sum(visit_counts.get((state, a), 0) for a in logprobs_dict)

    for a in logprobs_dict:
        log_p = logprobs_dict[a]
        n = visit_counts.get((state, a), 0)
        exploration_bonus = c * sqrt(log(total_visits + 1) / (n + 1))
        scores[a] = log_p + exploration_bonus
    return max(scores, key=scores.get), scores


def run(
    grid_size=6,
    obstacles={(2, 2), (3, 3), (1, 4)},
    image_path="data/grid.png",
    max_steps=30,
    prompt_mode="yesno",  # "multi" or "yesno"
):
    env = GridWorld(grid_size, obstacles=obstacles)
    env.initialize_agents_goals(num_agents=1)

    agent_pos = env.agents[0]
    goal_pos = env.goals[0]
    init_agent_pos = agent_pos

    step = 0
    visit_counts = {}
    llm_disagree_count = 0  # Counter for LLM vs UCG disagreement

    while agent_pos != goal_pos and step < max_steps:
        print(f"\n--- Step {step} ---")
        print(f"Agent position: {agent_pos}, Goal position: {goal_pos}")
        plot_grid(env, image_path=image_path)
        valid_actions = env.get_valid_actions(agent_pos)
        print(f"Valid actions: {valid_actions}")

        if prompt_mode == "multi":
            prompt = build_prompt_single_obs(agent_pos, goal_pos, valid_actions, grid_size, obstacles)
            sentence, logprobs = send_image_to_model_openai_logprobs(image_path, prompt, temperature=0.0000001)
            print(f"Raw logprobs: {logprobs}")
            logprobs_dict = extract_logprob_multichoice(logprobs, valid_actions)
            print(f"Filtered unnormalized logprobs: {logprobs_dict}")

        elif prompt_mode == "yesno":
            logprobs_dict = {}
            for action in valid_actions:
                prompt = build_yesno_prompt_single_obs(agent_pos, goal_pos, grid_size, obstacles, action)
                sentence, logprobs = send_image_to_model_openai_logprobs(image_path, prompt, temperature=0.0000001)
                print(f"Raw logprobs for '{action}': {logprobs}")
                logprobs_dict[action] = extract_logprob_yesno(logprobs)
            print(f"Filtered unnormalized logprobs: {logprobs_dict}")

        else:
            raise ValueError("Invalid prompt_mode")

        logprobs_dict = normalize_logprobs_to_logprobs(logprobs_dict)
        print(f"Normalized logprobs: {logprobs_dict}")

        # LLM suggestion: action with highest normalized logprob
        llm_action = max(logprobs_dict, key=logprobs_dict.get)
        direction, uct_scores = select_action_ucg(logprobs_dict, visit_counts, agent_pos)
        print(f"UCT scores: {uct_scores}")
        print(f"Selected action: {direction}")
        print(f"LLM suggested action: {llm_action}")

        if direction != llm_action:
            llm_disagree_count += 1
            print(f"Disagreement: UCG chose '{direction}', LLM chose '{llm_action}'")

        new_pos = env.move_agent(agent_pos, direction)
        print(f"Agent moves from {agent_pos} to {new_pos}")
        if new_pos == agent_pos:
            print("Agent did not move. Breaking loop.")
            break

        visit_counts[(new_pos, direction)] = visit_counts.get((new_pos, direction), 0) + 1
        agent_pos = new_pos
        env.agents[0] = agent_pos
        step += 1

    optimal = shortest_path_length(init_agent_pos, goal_pos, env)
    failed = step >= max_steps
    print(f"\nRun finished. Steps: {step}, Optimal: {optimal}, Failed: {failed}")
    print(f"LLM/UCG disagreement count: {llm_disagree_count}")
    return step, optimal, failed, llm_disagree_count


if __name__ == "__main__":
    steps, optimal, failed, llm_disagree_count = run()
    print(f"\n✅ Goal reached!")
    print(f"Optimal path length: {optimal}")
    print(f"Total steps taken  : {steps}")
    print(f"Failure            : {failed}")
    print(f"LLM/UCG disagreement : {llm_disagree_count}")
