from openai import OpenAI
import os
import json

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

model_name = 'chat_model_for_products'  

model_references = load_model_references(model_references_file)

# Get the model reference based on the model name
model_reference = get_model_reference(model_name, model_references)

if model_reference is None:
    raise ValueError(f"Model name '{model_name}' not found in the model references file.")


completion = client.chat.completions.create(
  model=model_reference,
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What can an Administrator on an unmanaged Deskvue Receiver do?"}
  ]
)
print(completion.choices[0].message)