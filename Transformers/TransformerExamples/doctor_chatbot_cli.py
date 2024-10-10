from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "johnos37/llama-3-8b-chat-doctor"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Chatbot is ready! Type 'exit' to stop.")
while True:
    input_text = input("You: ")
    if input_text.lower() == 'exit':
        break
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Chatbot: {response}")
