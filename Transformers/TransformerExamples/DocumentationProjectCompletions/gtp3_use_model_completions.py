from openai import OpenAI
import os
import json

# Set your API key

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],
)


def load_model_references(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    

def get_model_reference(model_name, references):
    return references.get(model_name)


current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


# Path to the JSON file containing model references
model_references_file = './config/models.json'

model_name = 'completions_model_for_products'  

model_references = load_model_references(model_references_file)

# Get the model reference based on the model name
model_reference = get_model_reference(model_name, model_references)

if model_reference is None:
    raise ValueError(f"Model name '{model_name}' not found in the model references file.")


def generate_completion(prompt, model):
    # Create a completion using the fine-tuned model
    response = client.completions.create(model=model,
    prompt=prompt,
    max_tokens=50)
    print(response)
    return response

# Example usage
model = model_reference
prompt = "Tell me about <product_name>"
completion = generate_completion(prompt, model)

# Print the response
print(completion.choices[0].text.strip())