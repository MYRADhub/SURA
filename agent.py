import ollama
from openai import OpenAI
import base64
import os
from dotenv import load_dotenv

def get_valid_actions(agent_pos, grid_size):
    row, col = agent_pos
    actions = []
    if row < grid_size - 1:
        actions.append("up")
    if row > 0:
        actions.append("down")
    if col > 0:
        actions.append("left")
    if col < grid_size - 1:
        actions.append("right")
    return actions

def build_prompt(agent_pos, target_pos, valid_actions, grid_size=8):
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

def send_image_to_model_ollama(image_path, agent_pos, target_pos, grid_size):
    valid_actions = get_valid_actions(agent_pos, grid_size)
    prompt = build_prompt(agent_pos, target_pos, valid_actions, grid_size=grid_size)
    
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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def send_image_to_model_openai(image_path, agent_pos, target_pos, grid_size):
    valid_actions = get_valid_actions(agent_pos, grid_size)
    prompt = build_prompt(agent_pos, target_pos, valid_actions)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    # Read and encode the image as base64
    base64_image = encode_image(image_path)

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
        temperature=0.0000001
    )
    return (prompt, response.output_text.strip().lower())

if __name__ == "__main__":
    image_path = "grid.png"
    response = send_image_to_model_ollama(image_path, (3, 4))
    print(f"Model response: {response}")