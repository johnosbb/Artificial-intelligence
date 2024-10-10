from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from datasets import load_dataset
import json
import torch
print(torch.cuda.is_available())

def load_token_references(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    

def get_token_reference(token_name, references):
    return references.get(token_name)


current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


# Path to the JSON file containing token references
token_references_file = './config/tokens.json'

hf_token_name = 'hf_Meta-Llama-3.1-8B-Instruct_RW'  
token_references = load_token_references(token_references_file)

# Get the model reference based on the model name
token_reference = get_token_reference(hf_token_name, token_references)

login(token = token_reference)

# Load the tokenizer and model
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define the input prompt
prompt = "Simplify the following text: When he woke in the woods in the dark and the cold of the night, he'd reach out to touch the child sleeping beside him."

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt")

# Move inputs to the same device as the model
inputs = {key: value.to(device) for key, value in inputs.items()}


# Generate text
outputs = model.generate(
    inputs["input_ids"],
    max_length=150,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
)

# Decode the generated output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the output
print("Generated Output:")
print(generated_text)
