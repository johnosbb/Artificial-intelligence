import re
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List


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


def classify_doc_type(file, text):
    filename = file.lower()

    # Check filename hints first
    if "software_design" in filename or "sdd" in filename:
        return "software design doc: "
    if "feature_description" in filename or "feature" in filename:
        return "feature description doc: "
    if "release_notes" in filename or "release" in filename or "RELEASE_NOTES" in filename:
        return "release note: "

    # Check for Document Type in markdown table format
    match = re.search(r'\|\s*Document Type:\s*\|\s*(.+?)\s*\|', text, re.IGNORECASE)
    if match:
        doc_type = match.group(1).strip().lower()
        if "software design" in doc_type:
            return "software design doc: "
        if "feature description" in doc_type:
            return "feature description doc: "
        if "release note" in doc_type:
            return "release note: "

    return "technical doc: "
