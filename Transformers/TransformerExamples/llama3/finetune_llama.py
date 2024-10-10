from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format
from huggingface_hub import login
import json


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

# wb_token = get_token_reference(wb_token_name, token_references)

# wandb.login(key=wb_token)
# run = wandb.init(
#     project='Fine-tune Llama 3 8B on Medical Dataset', 
#     job_type="training", 
#     anonymous="allow"
# )

base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
dataset_name = "ruslanmv/ai-medical-chatbot"
new_model = "llama-3-8b-chat-doctor"

torch_dtype = torch.float16
attn_implementation = "eager"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

# Adjust LLaMA configuration
config = LlamaConfig.from_pretrained(base_model)

# Correct the rope_scaling to include only 'type' and 'factor'
config.rope_scaling = {
    "type": "dynamic",  # Specify the type of scaling (dynamic or linear, depending on the use case)
    "factor": 8.0       # The scaling factor
}

# Load the model with modified configuration
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    config=config,  # Pass the corrected config with the simplified rope_scaling
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
model, tokenizer = setup_chat_format(model, tokenizer)


# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
model = get_peft_model(model, peft_config)

# Importing the dataset
dataset = load_dataset(dataset_name, split="all")
dataset = dataset.shuffle(seed=65).select(range(100)) # Only use 100 samples for quick demo

def format_chat_template(row):
    row_json = [{"role": "user", "content": row["Patient"]},
               {"role": "assistant", "content": row["Doctor"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc=4,
)

# Inspect formatted data
print(dataset['text'][3])

# Split the dataset into training and testing sets
dataset = dataset.train_test_split(test_size=0.1)

# Training arguments configuration
training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True
)

# Define the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    max_seq_length=512,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

# Train the model
trainer.train()

# Finish W&B logging
wandb.finish()

# Set use_cache to True
model.config.use_cache = True

# Example prompt for inference
messages = [
    {
        "role": "user",
        "content": "Hello doctor, I have bad acne. How do I get rid of it?"
    }
]

# Generate a response
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)

# Decode and print the output
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text.split("assistant")[1])

# Save and push the model to the Hugging Face Hub
trainer.model.save_pretrained(new_model)
trainer.model.push_to_hub(new_model, use_temp_dir=False)
