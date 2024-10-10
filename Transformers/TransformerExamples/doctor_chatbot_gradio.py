import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "johnos37/llama-3-8b-chat-doctor"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define a function to generate responses
def chat(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Create a Gradio interface
iface = gr.Interface(fn=chat, inputs="text", outputs="text", title="Llama 3 Chatbot")

# Launch the interface
iface.launch()
