import ollama
from openai import OpenAI
import base64
import os
from dotenv import load_dotenv

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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def send_image_to_model_openai(image_path, prompt, temperature=None):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    # Read and encode the image as base64
    base64_image = encode_image(image_path)

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
            temperature=temperature
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
            ]
        )
    return response.output_text.strip().lower()

if __name__ == "__main__":
    image_path = "grid.png"
    response = send_image_to_model_ollama(image_path, (3, 4))
    print(f"Model response: {response}")