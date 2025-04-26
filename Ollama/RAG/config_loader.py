import json
import os

CONFIG_PATH = os.environ.get("RAG_CONFIG_FILE", "config.json")  # Allow env override if needed

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

# Shortcut functions
_config = load_config()

def get_index_dir():
    return _config["INDEX_DIR"]

def get_document_folder_path():
    return _config["DOCUMENT_FOLDER_PATH"]

def get_output_dir():
    return _config["OUTPUT_DIRECTORY"]


def get(key, default=None):
    return _config.get(key, default)
