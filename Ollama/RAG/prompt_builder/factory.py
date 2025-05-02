from .internlm2 import InternLM2PromptBuilder
from .qwen25 import QwenPromptBuilder
# Extend here for other models

def get_prompt_builder(model_id: str):
    model_id = model_id.lower()

    if "internlm2" in model_id:
        return InternLM2PromptBuilder()
    elif "qwen" in model_id:
        return QwenPromptBuilder()
    # elif "mistral" in model_id:
    #     return MistralPromptBuilder()

    raise ValueError(f"Unsupported model ID: {model_id}")
