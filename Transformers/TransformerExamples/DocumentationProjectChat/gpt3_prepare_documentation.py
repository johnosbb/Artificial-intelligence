import os
import re
import nltk
import markdown
from bs4 import BeautifulSoup
import json
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# Function to convert Markdown to plain text
def convert_markdown_to_text(md_content):
    # Remove image references
    md_content = re.sub(r'!\[.*?\]\(.*?\)', '', md_content)
    
    # Remove code blocks
    md_content = re.sub(r'```.*?```', '', md_content, flags=re.DOTALL)
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content)
    
    # Use BeautifulSoup to extract plain text from HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    
    # Remove blank lines
    text = os.linesep.join([line for line in text.splitlines() if line.strip()])
    
    return text


def segment_text(text, max_tokens=2048):
    sentences = sent_tokenize(text)
    segments = []
    current_segment = []
    current_length = 0
    
    for sentence in sentences:
        token_count = len(nltk.word_tokenize(sentence))
        if current_length + token_count > max_tokens:
            segments.append(' '.join(current_segment))
            current_segment = [sentence]
            current_length = token_count
        else:
            current_segment.append(sentence)
            current_length += token_count
    
    if current_segment:
        segments.append(' '.join(current_segment))
    
    return segments



# Function to find Markdown files in a directory, excluding certain files
def find_markdown_files(directory, exclude_list):
    md_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md") and not any(exclude_word in file for exclude_word in exclude_list):
                md_files.append(os.path.join(root, file))
            else:
                print(f"Excluding file: {file}")    
    return md_files

# Function to concatenate text from all Markdown files
def concatenate_markdown_files(md_files):
    all_text = ""
    file_count = 0 
    for md_file in md_files:
        file_count += 1
        print(f"Processing file {file_count}: {md_file}")
        with open(md_file, 'r', encoding='utf-8') as file:
            md_content = file.read()
            text = convert_markdown_to_text(md_content)
            all_text += text + "\n"  # Add a newline to separate documents
    return all_text

# Function to write content to a file
def write_to_file(content, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(content)

# Function to prepare Markdown files for GPT-3 processing
def prepare_markdown_for_gpt3(input_directory, output_file_path, exclude_list):
    md_files = find_markdown_files(input_directory, exclude_list)
    concatenated_text = concatenate_markdown_files(md_files)
    write_to_file(concatenated_text, output_file_path)
    print(f"Concatenated text has been written to {output_file_path}")
    return concatenated_text

# Function to create prompts and completions for GPT-3 fine-tuning
def create_prompts_and_completions(text):
    sentences = sent_tokenize(text)
    prompt_separator = " ->"
    completion_ending = "\n"
    
    data = []
    for i in range(len(sentences) - 1):
        data.append({
            "prompt": sentences[i] + prompt_separator,
            "completion": " " + sentences[i + 1] + completion_ending
        })
    
    return data

def create_conversations(segments):
    conversations = []
    for segment in segments:
        conversation = {"messages": []}
        conversation["messages"].append({"role": "system", "content": "This is a conversation segment."})
        
        sentences = sent_tokenize(segment)
        for i in range(0, len(sentences) - 1, 2):
            user_content = sentences[i]
            assistant_content = sentences[i + 1]
            
            conversation["messages"].append({"role": "user", "content": user_content})
            conversation["messages"].append({"role": "assistant", "content": assistant_content})
        
        # Ensure the last message is from the assistant
        if len(sentences) % 2 == 1:
            last_message = {"role": "assistant", "content": sentences[-1]}
            if conversation["messages"][-1]["role"] == "assistant":
                conversation["messages"][-1]["content"] += " " + last_message["content"]
            else:
                conversation["messages"].append(last_message)
        
        conversations.append(conversation)
    
    return conversations

def save_conversations_to_jsonl(conversations, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for conversation in conversations:
            f.write(json.dumps(conversation) + '\n')

# Main function to execute the processing pipeline
def main():
    input_directory = "./data/md_docs/docs"
    output_file_path = "./data/md_docs/docs_chat.txt"
    exclude_list = ["RELEASE_NOTES", "release_notes", "tasklist"]
    json_output_file = './data/product_prompts_and_chat.jsonl'
    
    # Step 1: Prepare Markdown files
    concatenated_text = prepare_markdown_for_gpt3(input_directory, output_file_path, exclude_list)
    segments = segment_text(concatenated_text)
    conversations = create_conversations(segments)
    save_conversations_to_jsonl(conversations, json_output_file)


# Execute the main function
if __name__ == "__main__":
    main()
