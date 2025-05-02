from .base import PromptBuilder
from .modifiers import apply_prompt_modifiers

class QwenPromptBuilder(PromptBuilder):
    def build_prompt(self, model_id, docs, query, chat_history=None):
        system = (
            "You are a helpful assistant. Use ONLY the provided documents to answer the user's question."
        )

        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"=== DOCUMENTS ===\n{docs}\n\n"
            f"=== QUESTION ===\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        return apply_prompt_modifiers(prompt, query)
