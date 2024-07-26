import json
import openai
import os
from openai import OpenAI

# https://platform.openai.com/docs/guides/fine-tuning/create-a-fine-tuned-model

# # Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

UPLOAD=False
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],
)

def load_training_file_references(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    

def get_training_file_reference(training_file_name, references):
    return references.get(training_file_name)


current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


# Path to the JSON file containing model references
training_file_references_file = './config/training_files.json'

training_file_name = 'training_file_for_products'  

training_file_references = load_training_file_references(training_file_references_file)

# Get the model reference based on the model name
training_file_reference = get_training_file_reference(training_file_name, training_file_references)

if training_file_reference is None:
    raise ValueError(f"Model name '{training_file_name}' not found in the model references file.")


# from openai import OpenAI
# client = OpenAI()

if UPLOAD:
    #Our file needs to be uploaded to gpt3 using the Files API in order to be used with a fine-tuning jobs:

    jobID = client.files.create(
    file=open("./data/product_prompts_and_completions_prepared.jsonl", "rb"),
    purpose="fine-tune"
    )

    print("-------------------------------------------------\n")
    print(jobID)


model = client.fine_tuning.jobs.create(
  training_file=training_file_reference, 
  model="gpt-3.5-turbo"
)

print(model)