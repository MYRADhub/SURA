from core.environment import GridWorld
from core.plot import plot_grid
from core.prompt import (
    build_yesno_prompt_single_obs,
    # build_multichoice_prompt_single_obs
    build_prompt_single_obs
)
from core.agent import send_image_to_model_openai_logprobs
from math import sqrt, exp, log

def extract_logprob_multichoice(logprobs, valid_actions):
    """Extracts logprobs for valid actions from top_logprobs (first token only)."""
    logprob_dict = {}
    if not logprobs:
        return logprob_dict

    top_logprobs = logprobs[0].top_logprobs
    for item in top_logprobs:
        tok = item.token.strip().lower()
        if tok in valid_actions:
            logprob_dict[tok] = item.logprob
    return logprob_dict

def extract_logprob_yesno(logprobs):
    """Returns logprob for 'yes' token from top_logprobs."""
    if not logprobs:
        return float('-inf')

    for item in logprobs[0].top_logprobs:
        if item.token.strip().lower() == "yes":
            return item.logprob
    return float('-inf')

def normalize_logprobs_to_logprobs(logprobs_dict):
    """Converts raw logprobs to normalized logprobs (i.e., log softmax)"""
    max_logprob = max(logprobs_dict.values())  # for numerical stability
    exp_probs = {a: exp(lp - max_logprob) for a, lp in logprobs_dict.items()}
    total = sum(exp_probs.values())
    norm_probs = {a: p / total for a, p in exp_probs.items()}
    return {a: log(p) for a, p in norm_probs.items()}  # re-log after softmax

def select_action_ucg(logprobs_dict, visit_counts, state, c=1):
    scores = {}
    for a in logprobs_dict:
        log_p = logprobs_dict[a]              # model's confidence in log-prob
        n = visit_counts.get((state, a), 0)
        scores[a] = log_p + c * sqrt(1.0 / (n + 1))
    return max(scores, key=scores.get), scores


if __name__ == "__main__":
    grid_size = 6
    image_path = "data/grid.png"
    obstacles = {(2, 2), (3, 3), (1, 4)}
    env = GridWorld(grid_size, obstacles=obstacles)
    env.initialize_agents_goals(num_agents=1)

    agent_pos = env.agents[0]
    goal_pos = env.goals[0]
    init_agent_pos = agent_pos

    step = 0
    visit_counts = {}

    # Choose "multi" or "yesno"
    prompt_mode = "yesno"

    while agent_pos != goal_pos:
        print(f"\n--- Step {step} ---")
        print(f"Agent: {agent_pos}, Goal: {goal_pos}")

        plot_grid(env, image_path=image_path)
        valid_actions = env.get_valid_actions(agent_pos)

        if prompt_mode == "multi":
            prompt = build_prompt_single_obs(agent_pos, goal_pos, valid_actions, grid_size, obstacles)
            sentence, logprobs = send_image_to_model_openai_logprobs(image_path, prompt, temperature=0.0000001)
            print(f"Logprobs(returned): {logprobs}")
            print(f"Sentence: {sentence}")
            logprobs_dict = extract_logprob_multichoice(logprobs, valid_actions)

        elif prompt_mode == "yesno":
            logprobs_dict = {}
            for action in valid_actions:
                prompt = build_yesno_prompt_single_obs(agent_pos, goal_pos, grid_size, obstacles, action)
                sentence, logprobs = send_image_to_model_openai_logprobs(image_path, prompt, temperature=0.0000001)
                logprobs_dict[action] = extract_logprob_yesno(logprobs)

        else:
            raise ValueError("Invalid prompt_mode")

        print(f"Logprobs(raw): {logprobs_dict}")
        logprobs_dict = normalize_logprobs_to_logprobs(logprobs_dict)
        direction, score_dict = select_action_ucg(logprobs_dict, visit_counts, agent_pos, c=2)
        print(f"Selected: {direction}")
        print(f"Logprobs(normalized): {logprobs_dict}")
        print(f"UCT scores: {score_dict}")

        new_pos = env.move_agent(agent_pos, direction)
        if new_pos == agent_pos:
            print("Move blocked or invalid.")
            break
        state = new_pos
        action = direction
        visit_counts[(state, action)] = visit_counts.get((state, action), 0) + 10

        agent_pos = new_pos
        env.agents[0] = agent_pos
        step += 1

    print("\n✅ Goal reached!")
    optimal = abs(init_agent_pos[0] - goal_pos[0]) + abs(init_agent_pos[1] - goal_pos[1])
    print(f"Optimal path length: {optimal}")
    print(f"Total steps taken  : {step}")
