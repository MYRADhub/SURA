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
        "top_logprobs": 10,
        "max_tokens": 100
    }
    if temperature is not None:
        params["temperature"] = temperature
        params["top_p"] = 0.0000001

    response = client.chat.completions.create(**params)
    print('Received response')

    choices = response.choices
    logprobs = choices[0].logprobs.content
    sentence = choices[0].message.content
    sentence = sentence.lower()
    
    return sentence, logprobs

if __name__ == "__main__":
    image_path = "grid.png"
    response = send_image_to_model_ollama(image_path, (3, 4))
    print(f"Model response: {response}")