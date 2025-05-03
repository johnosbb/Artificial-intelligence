import re, os, requests
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from typing import List
import configparser
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote
import magic
from .document_types import DOC_TYPE_LABELS


class TextProcessingUtilities:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Downloading stopwords. Please run nltk.download('stopwords')")
            import nltk
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))

        self.DOC_TYPE_LABELS = DOC_TYPE_LABELS

    def extract_keywords(self, text, max_keywords=3):
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in self.stop_words and len(word) > 3]

        # Prioritize longer and more unique words
        keywords = sorted(set(keywords), key=lambda x: (-len(x), x))

        # Only take the top N keywords
        return keywords[:max_keywords]
    
    def get_filename_from_cd(self,cd):
        """
        Get filename from content-disposition
        """
        if not cd:
            return None
        fname = cd.split('filename=')[1]
        if fname.lower().startswith(("utf-8''", "utf-8'")):
            fname = fname.split("'")[-1]
        return unquote(fname)
    
    def download_file(self,url):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            filename = self.get_filename_from_cd(r.headers.get('content-disposition'))
            if not filename:
                filename = urlparse(url).geturl().replace('https://', '').replace('/', '-')
            filename = 'content/' + filename
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return filename

    def extract_release_version(self, text):
        """
        Try to extract the first version number (e.g., 1.4.0 or v1.4.0) from the text.
        Returns the first version found.
        """
        version_pattern = r"(v?\d+\.\d+\.\d+)"  # Pattern to match versions like 1.4.0 or v1.4.0

        # Search for the first matching version
        match = re.search(version_pattern, text)

        if match:
            return match.group(0)  # Return the first matched version

        return None  # Return None if no version is found
    
    def readtext(self,path):
        path = path.rstrip()
        path = path.replace(' \n', '')
        path = path.replace('%0A', '')
        if re.match(r'^https?://', path):
            filename = self.download_file(path)
        else:
            
            relative_path = path
            filename = os.path.abspath(relative_path)
        
        filetype = magic.from_file(filename, mime=True)
        print(f"\nEmbedding {filename} as {filetype}")
        text = ""
        if filetype == 'application/pdf':
            print('PDF not supported yet')
        if filetype == 'text/plain':
            with open(filename, 'rb') as f:
                text = f.read().decode('utf-8')
        if filetype == 'text/html':
            with open(filename, 'rb') as f:
                soup = BeautifulSoup(f, 'html.parser')
                text = soup.get_text()
        
        if os.path.exists(filename) and filename.find('content/') > -1:
            os.remove(filename) 
            
        return text

    def smart_chunk(self, text):
        structure = self.assess_document_structure(text)

        if structure == "bullet":
            return self.chunk_by_bullets(text, bullets_per_chunk=5)
        elif structure == "table":
            return self.chunk_by_tables(text)  # To be defined
        else:
            return self.chunk_by_sentences(text, sentences_per_chunk=3, overlap=1)  # Using a default for sentences_per_chunk and overlap

    def detect_language(self, file_path):
        if file_path.endswith('.py'):
            return 'python'
        if file_path.endswith(('.c', '.h')):
            return 'c'
        if file_path.endswith('.cpp'):
            return 'cpp'
        return 'unknown'

    def chunk_code_by_function(self, code_text):
        pattern = r'(def\s+\w+\s*\(.*?\):|[\w\s\*]+?\s+\**\w+\s*\(.*?\)\s*\{)'  # crude but catches Python and C defs
        lines = code_text.splitlines()

        chunks = []
        current_chunk = []
        in_function = False

        for line in lines:
            if re.match(pattern, line.strip()):
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                in_function = True
            if in_function:
                current_chunk.append(line)

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def chunk_by_tables(self, text):
        """
        Extracts markdown-style tables and returns them as individual chunks.
        Returns:
            List of table strings (each treated as a document chunk).
        """
        table_chunks = []

        # Pattern for markdown tables:
        # Captures a header row (line with pipes), followed by a separator (---), followed by 1+ rows
        table_pattern = re.compile(
            r"(?:^\|.+\|\n^\|[-| :]+\|\n(?:^\|.*\|\n?)+)",
            re.MULTILINE
        )

        matches = table_pattern.finditer(text)
        for match in matches:
            table = match.group(0).strip()
            table_chunks.append(table)

        return table_chunks

    def extract_section(self, text, heading="New functionality"):
        """Extracts a Markdown section following a heading like '### New functionality'."""
        # Match the heading line with optional prefixes (e.g., ###), ignoring case
        pattern = rf"^#+\s*{re.escape(heading)}.*?\n(.*?)(?=^#+\s|\Z)"  # Match until next heading or end
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        return match.group(1).strip() if match else None

    def extract_all_markdown_sections(self, text, heading="New functionality added/updated"):
        pattern = rf"^#+\s*{re.escape(heading)}.*?\n(.*?)(?=^#+\s|\Z)"
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
        return [m.strip() for m in matches]

    def chunk_by_bullets(self, text, bullets_per_chunk=5, overlap=0):
        lines = text.strip().splitlines()
        bullets = [line.strip() for line in lines if line.strip().startswith(("*", "-", "•", "◦"))]

        chunks = []
        for i in range(0, len(bullets), bullets_per_chunk - overlap):
            chunk = "\n".join(bullets[i:i + bullets_per_chunk])
            chunks.append(chunk)
        return chunks

    def chunk_by_sentences(self, source_text: str, sentences_per_chunk: int = 3, overlap: int = 1) -> List[str]:
        """
        Splits text by sentences with optional overlap.
        Default to 3 sentences per chunk with an overlap of 1.
        """
        if sentences_per_chunk < 2:
            raise ValueError("The number of sentences per chunk must be 2 or more.")
        if overlap < 0 or overlap >= sentences_per_chunk:
            raise ValueError("Overlap must be 0 or more and less than the number of sentences per chunk.")

        sentences = sent_tokenize(source_text)
        if not sentences:
            print("Nothing to chunk")
            return []

        chunks = []
        i = 0
        while i < len(sentences):
            end = min(i + sentences_per_chunk, len(sentences))
            chunk = ' '.join(sentences[i:end])

            if overlap > 0 and i > 0:
                overlap_start = max(0, i - overlap)
                overlap_end = i
                overlap_chunk = ' '.join(sentences[overlap_start:overlap_end])
                chunk = overlap_chunk + ' ' + chunk

            chunks.append(chunk.strip())
            i += sentences_per_chunk - overlap

        return chunks

    def get_config(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        return dict(config.items("main"))

    def assess_document_structure(self, text):
        lines = text.splitlines()
        bullet_count = sum(1 for line in lines if line.strip().startswith(("*", "-", "•", "◦")))
        table_row_count = sum(1 for line in lines if "|" in line and "---" not in line)
        total_lines = len(lines)

        bullet_ratio = bullet_count / total_lines if total_lines else 0
        table_ratio = table_row_count / total_lines if total_lines else 0

        if bullet_ratio > 0.3:
            return "bullet"
        elif table_ratio > 0.2:
            return "table"
        else:
            return "sentence"

    def is_table_heavy(self, text):
        return len(re.findall(r"^\|.+\|\n^\|[-| :]+\|", text, re.MULTILINE)) >= 1

    def is_bullet_heavy(self, text):
        return len(re.findall(r"^\s*[-*+•◦]\s", text, re.MULTILINE)) >= 3

    # ------------------ Type Classification ------------------

    def classify_doc_type(self, filename, text):
        filename = filename.lower()

        # First: Try to infer from filename
        for key, label in self.DOC_TYPE_LABELS.items():
            if key.replace(" ", "_") in filename or key in filename:
                print(f"[match] Detected '{label}' from filename")
                return label

        # Second: Try to extract from markdown metadata
        match = re.search(r'\|\s*Document Type:\s*\|\s*(.+?)\s*\|', text, re.IGNORECASE)
        if match:
            doc_type_text = match.group(1).strip().lower()
            for key, label in self.DOC_TYPE_LABELS.items():
                if key in doc_type_text:
                    print(f"[match] Detected '{label}' from document content")
                    return label

        # Default fallback
        print("[match] Defaulted to 'Technical Documents'")
        return "Technical Documents"

    def classify_code_component(self, file_path):
        """
        Try to classify code component based on file path or filename.
        E.g., detect 'driver', 'api', 'service', 'utility', etc.
        """
        file_path = file_path.lower()

        if "driver" in file_path:
            return "driver"
        if "api" in file_path:
            return "api"
        if "service" in file_path:
            return "service"
        if "util" in file_path or "utils" in file_path:
            return "utility"
        if "test" in file_path:
            return "test"
        if "core" in file_path:
            return "core module"

        # fallback
        return "miscellaneous"

# Example usage:
if __name__ == "__main__":
    text_utils = TextProcessingUtilities()
    sample_text = "This is a sample text about logging and version 1.5.0. It also mentions new functionality."
    keywords = text_utils.extract_keywords(sample_text)
    print(f"Keywords: {keywords}")
    version = text_utils.extract_release_version(sample_text)
    print(f"Release Version: {version}")

    bullet_text = """
    * Item 1
    * Item 2
    * Item 3
    * Item 4
    * Item 5
    * Item 6
    """
    bullet_chunks = text_utils.chunk_by_bullets(bullet_text, bullets_per_chunk=3)
    print(f"Bullet Chunks: {bullet_chunks}")

    sentence_text = "This is the first sentence. Here is the second sentence. And this is the third one. A fourth sentence follows. Finally, the fifth sentence."
    sentence_chunks = text_utils.chunk_by_sentences(sentence_text, sentences_per_chunk=2, overlap=1)
    print(f"Sentence Chunks: {sentence_chunks}")

    markdown_table = """
| Header 1 | Header 2 |
|---|---|
| Row 1 Col 1 | Row 1 Col 2 |
| Row 2 Col 1 | Row 2 Col 2 |
"""
    table_chunks = text_utils.chunk_by_tables(markdown_table)
    print(f"Table Chunks: {table_chunks}")

    doc_type_filename = "software_design_document.md"
    doc_type_content = "|| Document Type: | Software Design |"
    doc_type = text_utils.classify_doc_type(doc_type_filename, doc_type_content)
    print(f"Document Type: {doc_type}")