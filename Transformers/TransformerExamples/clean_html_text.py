import re
from bs4 import BeautifulSoup

# Open and read the downloaded HTML file
with open("./data/gutenberg.org_cache_epub_4280_pg4280.html", 'r', encoding='utf-8') as file:
    file_contents = file.read()

# Parse the file contents using BeautifulSoup
soup = BeautifulSoup(file_contents, 'html.parser')

# Get the text of the book and clean it up a bit
text = soup.get_text()
cleaned_text = re.sub('\s+', ' ', text).strip()

# Write the cleaned text to a new text file
output_file_path = "./data/cleaned_book_text.txt"
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write(cleaned_text)

print(f"Cleaned text has been written to {output_file_path}")
