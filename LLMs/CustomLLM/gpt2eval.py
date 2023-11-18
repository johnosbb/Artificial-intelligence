from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer
# You can specify other GPT-2 variants like "gpt2-medium", "gpt2-large", etc.
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Input text
input_text = "What is the best therapy for anxiety?"

# Encode the input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

attention_mask = torch.ones(input_ids.shape)

# Generate text
output = model.generate(input_ids, max_length=50,
                        num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
