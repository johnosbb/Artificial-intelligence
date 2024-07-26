import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

training_file_path = "../data/md_docs/docs.txt"

# Step 1: Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"  # You can also use "gpt2", "gpt2-medium", "gpt2-large", etc.
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Step 2: Prepare your corpus
dataset = load_dataset('text', data_files={'train': training_file_path})

# Step 3: Tokenize the dataset
max_length = 1024  # GPT-2's maximum token length

def tokenize_function(examples):
    return tokenizer(examples['text'], return_special_tokens_mask=True, truncation=True, max_length=max_length)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Step 4: Group the tokenized texts into chunks
block_size = 128

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_datasets = tokenized_datasets.map(group_texts, batched=True)

# Step 5: Set up training arguments and Trainer
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

# Step 6: Fine-tune the model
trainer.train()

# Step 7: Save the fine-tuned model
model.save_pretrained("../models/fine-tuned-gpt2")
tokenizer.save_pretrained("../models/fine-tuned-gpt2")

# # Import necessary libraries
# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
# from datasets import load_dataset


# training_file_path = "../data/md_docs/docs.txt"

# # Step 1: Load the pre-trained GPT-2 model and tokenizer
# model_name = "gpt2-large"  # You can also use "gpt2", "gpt2-medium", "gpt2-large", etc.
# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# # Step 2: Prepare your corpus
# # Assuming you have a text file `corpus.txt` with your product-related documents
# dataset = load_dataset('text', data_files={'train': '../data/md_docs/docs.txt'})

# # Step 3: Tokenize the dataset
# def tokenize_function(examples):
#     return tokenizer(examples['text'], return_special_tokens_mask=True)

# tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# # Step 4: Group the tokenized texts into chunks
# block_size = 128

# def group_texts(examples):
#     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     if total_length >= block_size:
#         total_length = (total_length // block_size) * block_size
#     result = {
#         k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
#         for k, t in concatenated_examples.items()
#     }
#     result["labels"] = result["input_ids"].copy()
#     return result

# tokenized_datasets = tokenized_datasets.map(group_texts, batched=True)

# # Step 5: Set up training arguments and Trainer
# training_args = TrainingArguments(
#     output_dir="./results",
#     overwrite_output_dir=True,
#     num_train_epochs=3,
#     per_device_train_batch_size=4,
#     save_steps=10_000,
#     save_total_limit=2,
#     prediction_loss_only=True,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets['train'],
# )

# # Step 6: Fine-tune the model
# trainer.train()

# # Step 7: Save the fine-tuned model
# model.save_pretrained("../models/fine-tuned-gpt2")
# tokenizer.save_pretrained("../models/fine-tuned-gpt2")
