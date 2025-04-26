import re
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List
from nltk.corpus import stopwords


def extract_keywords(text, max_keywords=3):
    stop_words = set(stopwords.words('english'))
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Prioritize longer and more unique words
    keywords = sorted(set(keywords), key=lambda x: (-len(x), x))
    
    # Only take the top N keywords
    return keywords[:max_keywords]


def extract_release_version(text):
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

def smart_chunk(text):
    structure = assess_document_structure(text)

    if structure == "bullet":
        return chunk_by_bullets(text, bullets_per_chunk=5)
    elif structure == "table":
        return chunk_by_tables(text)  # To be defined
    else:
        return chunk_by_sentences(text)  # Your existing method


def detect_language(file_path):
    if file_path.endswith('.py'):
        return 'python'
    if file_path.endswith(('.c', '.h')):
        return 'c'
    if file_path.endswith('.cpp'):
        return 'cpp'
    return 'unknown'



def chunk_code_by_function(code_text):
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


def chunk_by_tables(text):
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

def extract_section(text, heading="New functionality"):
    """Extracts a Markdown section following a heading like '### New functionality'."""
    # Match the heading line with optional prefixes (e.g., ###), ignoring case
    pattern = rf"^#+\s*{re.escape(heading)}.*?\n(.*?)(?=^#+\s|\Z)"  # Match until next heading or end
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
    return match.group(1).strip() if match else None



def extract_all_markdown_sections(text, heading="New functionality added/updated"):
    pattern = rf"^#+\s*{re.escape(heading)}.*?\n(.*?)(?=^#+\s|\Z)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    return [m.strip() for m in matches]


def chunk_by_bullets(text, bullets_per_chunk=5, overlap=0):
    lines = text.strip().splitlines()
    bullets = [line.strip() for line in lines if line.strip().startswith(("*", "-", "•", "•"))]
    
    chunks = []
    for i in range(0, len(bullets), bullets_per_chunk - overlap):
        chunk = "\n".join(bullets[i:i + bullets_per_chunk])
        chunks.append(chunk)
    return chunks

def chunk_by_sentences(source_text: str, sentences_per_chunk: int, overlap: int) -> List[str]:
    """
    Splits text by sentences
    """
    if sentences_per_chunk < 2:
        raise ValueError("The number of sentences per chunk must be 2 or more.")
    if overlap < 0 or overlap >= sentences_per_chunk - 1:
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
        
        if overlap > 0 and i > 1:
            overlap_start = max(0, i - overlap)
            overlap_end = i
            overlap_chunk = ' '.join(sentences[overlap_start:overlap_end])
            chunk = overlap_chunk + ' ' + chunk
        
        chunks.append(chunk.strip())
        i += sentences_per_chunk
    
    return chunks

# Could we do an initial assesment of a document to decide if it is
# Bullet-heavy → use chunk_by_bullets
# Table-heavy → extract full tables using regex or markdown parsing
# Otherwise → fallback to sentence-based chunking
def assess_document_structure(text):
    lines = text.splitlines()
    bullet_count = sum(1 for line in lines if line.strip().startswith(("*", "-", "•")))
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
    

def is_table_heavy(text):
    return len(re.findall(r"^\|.+\|\n^\|[-| :]+\|", text, re.MULTILINE)) >= 1

def is_bullet_heavy(text):
    return len(re.findall(r"^\s*[-*+•]\s", text, re.MULTILINE)) >= 3





# ------------------ Type Classification ------------------



DOC_TYPE_LABELS = {
    "software design": "Software Design Documents",
    "feature description": "Feature Description Documents",
    "release notes": "Release Notes",
    "architectural": "Architectural Documents",
    "technical": "Technical Documents",
    "performance analysis": "Performance Analysis Documents",
    "detailed product specification": "Detailed Product Specifications",
}

def classify_doc_type(filename, text):
    filename = filename.lower()

    # First: Try to infer from filename
    for key, label in DOC_TYPE_LABELS.items():
        if key.replace(" ", "_") in filename or key in filename:
            print(f"[match] Detected '{label}' from filename")
            return label

    # Second: Try to extract from markdown metadata
    match = re.search(r'\|\s*Document Type:\s*\|\s*(.+?)\s*\|', text, re.IGNORECASE)
    if match:
        doc_type_text = match.group(1).strip().lower()
        for key, label in DOC_TYPE_LABELS.items():
            if key in doc_type_text:
                print(f"[match] Detected '{label}' from document content")
                return label

    # Default fallback
    print("[match] Defaulted to 'Technical Documents'")
    return "Technical Documents"


def classify_code_component(file_path):
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
