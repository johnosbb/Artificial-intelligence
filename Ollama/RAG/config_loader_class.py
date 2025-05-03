import json
import os

class ConfigLoader:
    def __init__(self, config_path=None):
        self.config_path = config_path or os.environ.get("RAG_CONFIG_FILE", "config.json")
        self._config = self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {self.config_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.config_path}")
            return {}

    def get_index_dir(self):
        return self._config.get("INDEX_DIR")

    def get_document_folder_path(self):
        return self._config.get("DOCUMENT_FOLDER_PATH")

    def get_output_dir(self):
        return self._config.get("OUTPUT_DIRECTORY")

    def get(self, key, default=None):
        return self._config.get(key, default)

# Instantiate the config loader
config = ConfigLoader()

# Shortcut functions using the instance
def get_index_dir():
    return config.get_index_dir()

def get_document_folder_path():
    return config.get_document_folder_path()

def get_output_dir():
    return config.get_output_dir()

def getconfig():
    return config._config

def get(key, default=None):
    return config.get(key, default)

# Example usage (optional):
if __name__ == "__main__":
    print(f"Index Directory: {get_index_dir()}")
    print(f"Document Folder Path: {get_document_folder_path()}")
    print(f"Output Directory: {get_output_dir()}")
    print(f"Main Model from Config: {get('mainmodel')}")
    print(f"Non-existent Key: {get('non_existent_key', 'default_value')}")