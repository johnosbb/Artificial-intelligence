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
  training_file="file-xx", 
  model="babbage-002"
)

print(model)