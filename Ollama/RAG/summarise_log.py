#!/usr/bin/env python3
import os
import sys
import time
import re
import argparse
import chromadb
import ollama
import nltk

from rag_utilities_class import TextProcessingUtilities


class LogAnalyser:
    def __init__(self):
        self.ru = TextProcessingUtilities()
        self.main_model = self.ru.get_config()["mainmodel"]


    def generate_summary(self, text):
        # prompt = (
        #     "You are given a logfile.\n"
        #     "Analyze the log contents and create a report with these sections:\n"
        #     "- A Summary\n"
        #     "- Key Findings\n"
        #     "- Network Performance Analysis\n"
        #     "- System Health Status\n"
        #     "- Trends and Notable Patterns within the data\n\n"
        #     f"=== LOGFILE CONTENT ===\n{text}\n=== END LOGFILE CONTENT ===\n\n"
        #     "Final consolidated summary:"
        # )
        prompt = (
            "You are a network engineer.\n"
            "Analyze the log contents below and summarise  \n"
            " the timeline of any critical events and\n"
            " any trends and notable Patterns within the data\n\n"
            f"=== LOGFILE CONTENT ===\n{text}\n=== END LOGFILE CONTENT ===\n\n"
        )
        response = ollama.generate(model=self.main_model, prompt=prompt)
        return response["response"].strip()
    
    def read_input(self,path: str) -> str:
        if path == "-":
            return sys.stdin.read()
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        except PermissionError:
            print(f"Error: permission denied: {path}", file=sys.stderr)
            sys.exit(1)


def main() -> int:
    logAnalyser = LogAnalyser()
    parser = argparse.ArgumentParser(
        description="Summarize a logfile using a local Ollama model."
    )
    parser.add_argument(
        "logfile",
        help="Path to the logfile to summarize, or '-' to read from stdin.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model to use. Overrides configuration sources.",
    )
    args = parser.parse_args()

    text = logAnalyser.read_input(args.logfile).strip()
    if not text:
        print("Error: input is empty; nothing to summarize.", file=sys.stderr)
        return 1

    summary = logAnalyser.generate_summary( text)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())

