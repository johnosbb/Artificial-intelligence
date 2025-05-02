from .base import PromptBuilder
from .modifiers import apply_prompt_modifiers

class InternLM2PromptBuilder(PromptBuilder):
    def build_prompt(self, model_id, docs, query, chat_history=None):
        # Static system message for your RAG-style task
        system = (
            "You are a helpful assistant. ONLY use the provided documents to summarise the user's requested release. "
            "Provide a bullet-point list of the top 10 most important changes. If the answer is not in the documents, say so."
        )

        prompt = (
            f"<|im_start|>system\n{system}\n<|im_end|>\n"
            f"<|im_start|>user\n"
            f"=== DOCUMENTS ===\n{docs}\n\n"
            f"=== QUESTION ===\n{query}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        return apply_prompt_modifiers(prompt, query)
