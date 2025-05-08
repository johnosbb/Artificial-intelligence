#!/usr/bin/env python3

import re
import sys

VALID_TYPES = [
    "software design document",
    "feature description document",
    "release notes"
]

EXAMPLE_BLOCK = """
🔧 Example of a valid Document Type section in a markdown table:

## Document Properties

| Property        | Value                    |
|----------------|--------------------------|
| Document Type: | Software Design Document |
"""

def is_valid_doc(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"⚠️ Could not read {file_path}: {e}")
        return False

    # Match something like: | Document Type: | Software Design Document |
    match = re.search(r'\|\s*Document Type:\s*\|\s*(.+?)\s*\|', text, re.IGNORECASE)
    if match:
        doc_type = match.group(1).strip().lower()
        return doc_type in VALID_TYPES
    return False

def main():
    files = sys.argv[1:]
    invalid_files = []
    for file in files:
        if file.endswith(".md") and not is_valid_doc(file):
            invalid_files.append(file)

    if invalid_files:
        print("❌ Commit blocked. The following markdown files are missing or have invalid 'Document Type' entries:")
        for f in invalid_files:
            print(f"  - {f}")
        print("\n✅ Valid types are:")
        for t in VALID_TYPES:
            print(f"  - {t}")
        print(EXAMPLE_BLOCK)
        sys.exit(1)

if __name__ == "__main__":
    main()
