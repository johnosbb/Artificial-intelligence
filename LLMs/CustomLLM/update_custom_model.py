from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os

# Load the existing fine-tuned model and tokenizer
# Load your existing fine-tuned model
model_name = "./custom_models/"
tokenizer = GPT2Tokenizer.from_pretrained(
    model_name)  # Use from_pretrained with tokenizer
model = GPT2LMHeadModel.from_pretrained(model_name)  # Load the model

# Directory containing new data
new_data_dir = "/home/johnos/DataSets/Gutenberg/textfiles/"

# List new text files in the directory
new_text_files = [file for file in os.listdir(
    new_data_dir) if file.endswith(".txt")]

# Initialize variables to hold combined data
combined_text = ""

# Loop through new text files, read and concatenate their content
for file in new_text_files:
    with open(os.path.join(new_data_dir, file), "r", encoding="utf-8") as f:
        text = f.read()
        combined_text += text

# Tokenize the combined new data
input_ids = tokenizer.encode(combined_text, return_tensors="pt",
                             max_length=128, truncation=True, padding="max_length")

# Create a dataset that combines old and new data
combined_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=input_ids,
    block_size=128  # Adjust this based on your dataset and model requirements
)

# Create a data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    save_steps=10_000,
)

# Create a Trainer and fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=combined_dataset,
)

# Fine-tune the model with the combined dataset
trainer.train()

# Save the updated model
trainer.save_model()
