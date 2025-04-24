#!/usr/bin/env python3
import os
import argparse
from datetime import date

def insert_document_properties(file_path, document_type):
    """Inserts a 'Document Properties' section into a Markdown file.

    The section is inserted after the first top-level heading (starting with '# ').
    If the file already contains a '## Document Properties' section, it is skipped.
    If no top-level heading is found, the file is skipped.

    Args:
        file_path (str): The path to the Markdown file.
        document_type (str): The value to be inserted for 'Document Type'.
    """
    today = date.today().strftime("%d/%m/%Y")
    document_properties_section = (
        "\n"
        "## Document Properties\n\n"
        "|                    |                                   |\n"
        "| ------------------ | --------------------------------- |\n"
        "| Document Number:   | Add a unique document number here |\n"
        f"| Document Type:     | {document_type}                     |\n"
        "| Author:            | add your name here                |\n"
        "| Status:            | Pending                           |\n"
        "| Rev:               | 1.0.0                             |\n"
        f"| Date:              | {today}                             |\n"
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
    """Checks if a filename ends with the '.md' extension."""
    return filename.endswith('.md')

def main():
    """Main function to parse arguments and process Markdown files."""
    parser = argparse.ArgumentParser(
        description="Insert a 'Document Properties' section into Markdown files within a specified folder.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Example Usage:\n"
               "  insert_properties.py ./my_documents Specification\n"
               "  insert_properties.py /path/to/reports Report"
    )
    parser.add_argument("folder_location", help="Path to the folder containing the Markdown files.")
    parser.add_argument("document_type", help="The type of document to be inserted in the properties (e.g., 'Specification', 'Report').")

    args = parser.parse_args()

    if not os.path.isdir(args.folder_location):
        print(f"Error: Folder not found at '{args.folder_location}'")
        return

    print(f"Processing Markdown files in: {args.folder_location}")
    for root, dirs, files in os.walk(args.folder_location):
        for file in files:
            if is_markdown_file(file):
                file_path = os.path.join(root, file)
                insert_document_properties(file_path, args.document_type)

if __name__ == "__main__":
    main()