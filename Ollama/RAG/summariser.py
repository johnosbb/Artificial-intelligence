#!/usr/bin/env python3
import os
import sys
import time
import argparse
import ollama



# # Text from file, prompt from file
# ./summariser.py ./logs/system.log ./prompts/network_prompt.txt

# # Text from stdin, prompt from file
# cat ./logs/system.log | ./summariser.py - ./prompts/network_prompt.txt

# # Text from file, prompt from stdin
# cat ./prompts/network_prompt.txt | ./summariser.py ./logs/system.log -

# # With a specific model
# ./summariser.py ./logs/system.log ./prompts/network_prompt.txt --model llama3.1:8b


from rag_utilities_class import TextProcessingUtilities


class Summariser:
    def __init__(self):
        self.ru = TextProcessingUtilities()
        self.main_model = self.ru.get_config()["mainmodel"]

    def set_model(self, model_name: str | None) -> None:
        if model_name:
            self.main_model = model_name

    def generate_summary(self, text: str, prompt: str) -> str:
        """
        Generate a summary using the provided prompt and text.
        The prompt should contain whatever instructions are desired.
        The text is appended after the prompt as TEXTFILE CONTENT.
        """
        full_prompt = (
            f"{prompt.rstrip()}\n\n"
            f"=== TEXTFILE CONTENT ===\n{text}\n=== END TEXTFILE CONTENT ===\n"
        )
        response = ollama.generate(model=self.main_model, prompt=full_prompt)
        return response["response"].strip()

    def read_input(self, path: str) -> str:
        """
        Read text content from a path. If path is '-', read from stdin.
        """
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


def _timestamp() -> str:
    # Local time; safe for filenames
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def _write_summary_file(summary_text: str) -> str:
    out_name = f"summary_{_timestamp()}.txt"
    try:
        with open(out_name, "w", encoding="utf-8") as f:
            f.write(summary_text)
    except OSError as e:
        print(f"Error: could not write output file '{out_name}': {e}", file=sys.stderr)
        sys.exit(1)
    return out_name


def main() -> int:
    summariser = Summariser()
    parser = argparse.ArgumentParser(
        description="Summarize text using a local Ollama model with a separate prompt file."
    )
    parser.add_argument(
        "textfile",
        help="Path to the text to summarize, or '-' to read from stdin.",
    )
    parser.add_argument(
        "promptfile",
        help="Path to a text file containing the prompt. Use '-' to read from stdin (cannot be '-' if textfile is also '-').",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model to use. Overrides configuration sources.",
    )
    args = parser.parse_args()

    # Prevent reading both inputs from stdin simultaneously
    if args.textfile == "-" and args.promptfile == "-":
        print("Error: textfile and promptfile cannot both be '-'. Provide at least one file path.", file=sys.stderr)
        return 1

    # Apply model override if provided
    summariser.set_model(args.model)

    # Read inputs
    text = summariser.read_input(args.textfile).strip()
    if not text:
        print("Error: input text is empty; nothing to summarize.", file=sys.stderr)
        return 1

    prompt = summariser.read_input(args.promptfile).strip()
    if not prompt:
        print("Error: prompt is empty; provide a non-empty prompt file.", file=sys.stderr)
        return 1

    # Generate and write summary
    summary = summariser.generate_summary(text, prompt)
    out_path = _write_summary_file(summary)
    print(f"Wrote summary to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
