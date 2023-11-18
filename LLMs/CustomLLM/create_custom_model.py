from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


data_path = "/home/johnos/DataSets/Gutenberg/textfiles/Frankenstein_by_Mary_Shelley.rtf.txt"
# Prepare your fine-tuning dataset and dataloaders
block_size = 128

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=data_path,
    block_size=block_size
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    save_steps=10_000,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()
trainer.save_model()
