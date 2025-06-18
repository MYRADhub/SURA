import ollama
from openai import OpenAI
import base64
import os
from dotenv import load_dotenv
from core.schema import OpenAIResponse

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def send_image_to_model_ollama(image_path, prompt):
    
    response = ollama.chat(
        model='llava',
        messages=[
            {
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }
        ]
    )
    return (prompt, response['message']['content'].strip().lower())

def send_image_to_model_openai_responses_api(image_path, prompt, temperature=None):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    # Read and encode the image as base64
    base64_image = encode_image(image_path)

    print('Sending request to OpenAI API')
    if temperature:
        response = client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": prompt },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}",
                        },
                    ]
                }
            ],
            temperature=temperature,
            top_p=0.0000001,
            max_output_tokens=100
        )
    else:
        response = client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": prompt },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}",
                        },
                    ]
                }
            ],
            max_output_tokens=100
        )
    print('Received response')
    return response.output_text.strip().lower()

def send_image_to_model_openai(image_path, prompt, temperature=None):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    # Read and encode the image as base64
    base64_image = encode_image(image_path)

    print('Sending request to OpenAI API')
    
    params = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ]
            }
        ],
        "max_tokens": 100
    }
    
    if temperature is not None:
        params["temperature"] = temperature
        params["top_p"] = 0.0000001

    response = client.chat.completions.create(**params)
    print('Received response')
    return response.choices[0].message.content.strip().lower()

def send_image_to_model_openai_logprobs(image_path, prompt, temperature=None):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    # Read and encode the image as base64
    base64_image = encode_image(image_path)

    print('Sending request to OpenAI API')

    params = {
        "model": "gpt-4.1",
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ]
            }
        ],
        "logprobs": True,
        "top_logprobs": 10,
        # "max_tokens": 300
    }
    if temperature is not None:
        params["temperature"] = temperature
        # params["top_p"] = 0.0000001

    response = client.chat.completions.create(**params)
    print('Received response')

    choices = response.choices
    logprobs = choices[0].logprobs.content
    sentence = choices[0].message.content
    
    return sentence, logprobs

def send_image_to_model_openai_logprobs_formatted(image_path, prompt, temperature=None):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    # Read and encode the image as base64
    base64_image = encode_image(image_path)
    print('Sending request to OpenAI API')
    params = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ]
            }
        ],
        "logprobs": True,
        "top_logprobs": 3,
        "max_tokens": 100,
        "response_format": OpenAIResponse
    }

    if temperature is not None:
        params["temperature"] = temperature
        params["top_p"] = 0.0000001

    response = client.beta.chat.completions.parse(**params)
    print('Received response')
    choices = response.choices
    logprobs = choices[0].logprobs.content
    sentence = choices[0].message.content
    
    return sentence, logprobs

def build_test_prompt():
    # Example values for testing
    agent_id = 1
    agent_pos = (2, 3)
    goal_positions = [(0, 0), (4, 4), None]
    other_agents = [(2, (1, 1))]
    grid_size = 5
    obstacles = [(1, 2), (3, 3)]
    direction = "up"
    memory = [(2, 3, "left", 2, 2), (2, 2, "up", 3, 2)]
    visits = {(2, 3): 2, (3, 2): 1, (2, 2): 3, (3, 3): 0}

    # Build a prompt similar to build_yesno_prompt_unassigned_goals
    obs_coords = ', '.join([f"({r}, {c})" for r, c in sorted(obstacles)])
    if memory:
        history_lines = "\n".join(
            [f"  • {i+1}. you moved from (row {r0}, col {c0}) **{dir_}** → got to (row {r1}, col {c1})"
             for i, (r0, c0, dir_, r1, c1) in enumerate(memory[:5])]
        )
    else:
        history_lines = "  • (no prior moves — this is the first step)"
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
    if other_agents:
        other_agent_lines = "\n".join(
            [f"  • Agent {aid} is at (row {pos[0]}, col {pos[1]})" for aid, pos in other_agents]
        )
    else:
        other_agent_lines = "  • (no other agents present)"
    goal_lines = "\n".join(
        [f"  • Goal {chr(65+i)} is at (row {pos[0]}, col {pos[1]})" for i, pos in enumerate(goal_positions) if pos is not None]
    )
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

if __name__ == "__main__":
    image_path = "data/grid.png"
    # response = send_image_to_model_ollama(image_path, (3, 4))
    prompt = build_test_prompt()
    print(f"Prompt: {prompt}")
    response = send_image_to_model_openai_logprobs_formatted(image_path, prompt, temperature=0.0000001)
    print(f"Model response: {response}")