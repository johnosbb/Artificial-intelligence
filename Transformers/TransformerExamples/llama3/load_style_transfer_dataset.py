import pandas as pd
from huggingface_hub import login
from datasets import load_dataset, Dataset
import os
import json

def load_token_references(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    

def get_token_reference(token_name, references):
    return references.get(token_name)

# Path to the JSON file containing token references
token_references_file = './config/tokens.json'

current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


hf_token_name = 'hf_Meta-Llama-3.1-8B-Instruct_RW'  
token_references = load_token_references(token_references_file)


# splits = {'train': 'train.tsv', 'validation': 'val.tsv'}
# df = pd.read_csv("hf://datasets/jdpressman/retro-text-style-transfer-v0.1/" + splits["train", sep="\t"])



ds = load_dataset("jdpressman/retro-text-style-transfer-v0.1")

print(ds.keys())

# Access the training split
train_data = ds['train']

# Check the first few examples in the training split
print(train_data[0])  # Print the first example

# View the first 5 samples in the training set
for i in range(5):
    print(f"Example {i + 1}:")
    print(train_data[i])  # Display the ith example
    print("---")


# Check the features of the training split
print(f"Features:\n {train_data.features}")


# Access the training split
train_data = ds['train']

# Convert to a list of dictionaries
train_data_list = train_data.to_list()

# Specify the filename
filename = './data/train_data.json'

# Write the data to a JSON file
with open(filename, 'w') as json_file:
    json.dump(train_data_list, json_file, indent=4)

