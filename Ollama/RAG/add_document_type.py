#!/usr/bin/env python3
import os
import argparse
from datetime import date

def insert_document_properties(file_path, document_type):
    today = date.today().strftime("%d/%m/%Y")
    document_properties_section = (
        "\n"
        "## Document Properties\n\n"
        "|                  |                                   |\n"
        "| ---------------- | --------------------------------- |\n"
        "| Document Number: | Add a unique document number here |\n"
        f"| Document Type:   | {document_type} |\n"
        "| Author:          | add your name here                |\n"
        "| Status:          | Pending                           |\n"
        "| Rev:             | 1.0.0                             |\n"
        f"| Date:            | {today}                           |\n"
        "\n"
    )

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        content = ''.join(lines)

    if "## Document Properties" in content:
        print(f"Skipped (already has 'Document Properties'): {file_path}")
        return

    new_lines = []
    inserted = False
    for line in lines:
        new_lines.append(line)
        if not inserted and line.strip().startswith("# "):
            new_lines.append(document_properties_section)
            inserted = True

    if inserted:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(new_lines)
        print(f"Updated: {file_path}")
    else:
        print(f"Skipped (no top-level heading): {file_path}")

def is_markdown_file(filename):
    return filename.endswith('.md')

def main():
    parser = argparse.ArgumentParser(description="Insert a Document Properties section into Markdown files.")
    parser.add_argument("folder_location", help="Path to the folder containing the Markdown files")
    parser.add_argument("document_type", help="Value for the 'Document Type' field")

    args = parser.parse_args()

    for root, dirs, files in os.walk(args.folder_location):
        for file in files:
            if is_markdown_file(file):
                insert_document_properties(os.path.join(root, file), args.document_type)

if __name__ == "__main__":
    main()
