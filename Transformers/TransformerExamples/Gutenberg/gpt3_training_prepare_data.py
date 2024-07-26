#From Gutenberg to JSON
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import requests
from bs4 import BeautifulSoup
import json
import re


# First, fetch the text of the book
# Option 1: from Project Gutenberg
#url = 'http://www.gutenberg.org/cache/epub/4280/pg4280.html'
#response = requests.get(url)
#soup = BeautifulSoup(response.content, 'html.parser')

# Option 2: from the GitHub repository:
#Development access to delete when going into production
#!curl -H 'Authorization: token {github_token}' -L https://raw.githubusercontent.com/Denis2054/Transformers-for-NLP-and-Computer-Vision-3rd-Edition/master/Chapter08/gutenberg.org_cache_epub_4280_pg4280.html --output "gutenberg.org_cache_epub_4280_pg4280.html"

# Open and read the downloaded HTML file
with open("./data/mv ../.org_cache_epub_4280_pg4280.html", 'r', encoding='utf-8') as file:
    file_contents = file.read()

# Parse the file contents using BeautifulSoup
soup = BeautifulSoup(file_contents, 'html.parser')

# Get the text of the book and clean it up a bit
text = soup.get_text()
text = re.sub('\s+', ' ', text).strip()



# Split the text into sentences
sentences = sent_tokenize(text)

# Define the separator and ending
prompt_separator = " ->"
completion_ending = "\n"

# Now create the prompts and completions
data = []
for i in range(len(sentences) - 1):
    data.append({
        "prompt": sentences[i] + prompt_separator,
        "completion": " " + sentences[i + 1] + completion_ending
    })

# Write the prompts and completions to a file
with open('./data/kant_prompts_and_completions.json', 'w') as f:
    for line in data:
        f.write(json.dumps(line) + '\n')