import json
from openai import OpenAI

client = OpenAI()
from openai import OpenAI
import os

# Open the file and read the lines
with open('./data/kant_prompts_and_completions_prepared.jsonl', 'r') as f:
    lines = f.readlines()

# Parse and print the first 5 lines
for line in lines[199:300]:
    data = json.loads(line)
    print(json.dumps(data, indent=4))


# Get API key from environment
my_api_key = os.getenv('OPENAI_API_KEY')

# Print the API key for verification (ensure this is removed in production for security)
print(f"API_KEY : {my_api_key}")



client = OpenAI(
    # This is the default and can be omitted
    api_key=my_api_key,
)





print(client.fine_tuning.jobs.list(limit=10))



fine_tuned_model = "ft:babbage-002:personal::xxxxx"

# Create a completion using the fine-tuned model
response = client.completions.create(model=fine_tuned_model,
prompt="Several concepts are a priori such as",
max_tokens=50)
print(response)



