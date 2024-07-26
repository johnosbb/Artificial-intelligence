import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model_path = "../models/fine-tuned-gpt2"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Function to generate answers
def generate_answer(question, model, tokenizer, max_new_tokens=50,max_length=150, num_return_sequences=1):
    # Encode the question
    input_ids = tokenizer.encode(question, return_tensors='pt')

    # Ensure the input length does not exceed max_length
    if input_ids.shape[1] > max_length:
        input_ids = input_ids[:, -max_length:]
    
    # Create attention mask
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    
    # Generate the answer using the model
    output = model.generate(
        input_ids, 
        max_new_tokens=max_new_tokens,
        # max_length=max_length, 
        num_return_sequences=num_return_sequences, 
        no_repeat_ngram_size=2, 
        top_k=50, 
        top_p=0.95, 
        temperature=0.7,  # Set temperature if do_sample=True
        do_sample=True,  # Enable sampling
        attention_mask=attention_mask,  # Set attention mask
        pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id to eos_token_id
        eos_token_id=tokenizer.eos_token_id  # Set eos_token_id
    )
    
    # Decode the generated answer
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    # Post-process to ensure the text ends at the end of a sentence
    end_punctuations = {'.', '!', '?'}
    for i in range(len(answer) - 1, -1, -1):
        if answer[i] in end_punctuations:
            answer = answer[:i + 1]
            break
    
    return answer

# # Example usage
# question = "Can you tell me about Kants understanding of the divine?"
# answer = generate_answer(question, model, tokenizer)
# print("Question:", question)
# print("Answer:", answer)

# Interactive chat loop
chat_history = ""
max_history_tokens = 1024  # Set the maximum token limit for history


print("Start chatting with the model (type 'exit' to stop):")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    # Append user input to chat history
    chat_history += f"You: {user_input}\n"
    
    # Generate response
    response = generate_answer(chat_history, model, tokenizer, max_new_tokens=50, max_length=max_history_tokens)
    
    # Append model response to chat history
    chat_history += f"Model: {response}\n"
    
    print(f"Model: {response}")