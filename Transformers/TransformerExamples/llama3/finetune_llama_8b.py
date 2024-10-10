# Installing More Dependencies
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import os
import json
from huggingface_hub import login


model_id="meta-llama/Meta-Llama-3.1-8B"

def load_token_references(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    

def get_token_reference(token_name, references):
    return references.get(token_name)


current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


# Path to the JSON file containing token references
token_references_file = './config/tokens.json'

hf_token_name = 'hf_Meta-Llama-3.1-8B-Instruct'  
wb_token_name = 'wanb' 
token_references = load_token_references(token_references_file)

# Get the model reference based on the model name
token_reference = get_token_reference(hf_token_name, token_references)

login(token = token_reference)


def get_model_and_tokenizer(model_id):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache=False
    model.config.pretraining_tp=1
    return model, tokenizer

model, tokenizer = get_model_and_tokenizer(model_id)

from transformers import GenerationConfig
from time import perf_counter

def generate_response(user_input):

  prompt = formatted_prompt(user_input)

  inputs = tokenizer([prompt], return_tensors="pt")
  generation_config = GenerationConfig(penalty_alpha=0.6,do_sample = True,
      top_k=5,temperature=0.5,repetition_penalty=1.2,
      max_new_tokens=60,pad_token_id=tokenizer.eos_token_id
  )
  start_time = perf_counter()

  inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

  outputs = model.generate(**inputs, generation_config=generation_config)
  theresponse = (tokenizer.decode(outputs[0], skip_special_tokens=True))
  print(tokenizer.decode(outputs[0], skip_special_tokens=True))
  output_time = perf_counter() - start_time
  print(f"Time taken for inference: {round(output_time,2)} seconds")
  #return theresponse

  def formatted_prompt(question)-> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant:"
  
  