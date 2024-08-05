# Use a pipeline as a high-level helper
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import os
import json

# https://www.datacamp.com/tutorial/llama3-fine-tuning-locally

def load_token_references(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    

def get_token_reference(token_name, references):
    return references.get(token_name)


current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


# Path to the JSON file containing token references
token_references_file = './config/tokens.json'

token_name = 'hf_Meta-Llama-3.1-8B-Instruct'  

token_references = load_token_references(token_references_file)

# Get the model reference based on the model name
token_reference = get_token_reference(token_name, token_references)

if token_reference is None:
    raise ValueError(f"Model name '{token_name}' not found in the model references file.")

print(f"Logging in using token reference: {token_reference}")
login(token = token_reference)

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)



messages = [
    {"role": "system", "content": "You are Virginnina Woolfe"},
    {"role": "user", "content": "What is it to live a complete life?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])