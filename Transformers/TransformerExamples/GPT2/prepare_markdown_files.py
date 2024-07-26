import os
import re
import markdown
from bs4 import BeautifulSoup

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

def find_markdown_files(directory, exclude_list):
    md_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md") and not any(exclude_word in file for exclude_word in exclude_list):
                md_files.append(os.path.join(root, file))
            else:
                print(f"Excluding file: {file}")    
    return md_files

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

def write_to_file(content, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(content)

def prepare_markdown_for_gpt2(input_directory, output_file_path, exclude_list):
    md_files = find_markdown_files(input_directory, exclude_list)
    concatenated_text = concatenate_markdown_files(md_files)
    write_to_file(concatenated_text, output_file_path)
    print(f"Concatenated text has been written to {output_file_path}")

# Example usage
input_directory = "./data/md_docs/docs"
output_file_path = "./data/md_docs/docs.txt"
exclude_list = ["RELEASE_NOTES", "release_notes","tasklist"]
prepare_markdown_for_gpt2(input_directory, output_file_path, exclude_list)


# import os
# import markdown
# from bs4 import BeautifulSoup

# def convert_markdown_to_text(md_content):
#     # Convert markdown to HTML
#     html_content = markdown.markdown(md_content)
#     # Use BeautifulSoup to extract plain text from HTML
#     soup = BeautifulSoup(html_content, 'html.parser')
#     text = soup.get_text()
#     return text

# def find_markdown_files(directory):
#     md_files = []
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.endswith(".md"):
#                 md_files.append(os.path.join(root, file))
#     return md_files

# def concatenate_markdown_files(md_files):
#     all_text = ""
#     file_count = 0 
#     for md_file in md_files:
#         file_count = file_count +1
#         print(f"Processing file {file_count}: {md_file}")
#         with open(md_file, 'r', encoding='utf-8') as file:
#             md_content = file.read()
#             text = convert_markdown_to_text(md_content)
#             all_text += text + "\n"  # Add a newline to separate documents
#     return all_text

# def write_to_file(content, output_file_path):
#     with open(output_file_path, 'w', encoding='utf-8') as output_file:
#         output_file.write(content)

# def prepare_markdown_for_gpt2(input_directory, output_file_path):
#     md_files = find_markdown_files(input_directory)
#     concatenated_text = concatenate_markdown_files(md_files)
#     write_to_file(concatenated_text, output_file_path)
#     print(f"Concatenated text has been written to {output_file_path}")

# # Example usage
# input_directory = "./data/md_docs/docs"
# output_file_path = "./data/md_docs/docs.txt"
# prepare_markdown_for_gpt2(input_directory, output_file_path)
