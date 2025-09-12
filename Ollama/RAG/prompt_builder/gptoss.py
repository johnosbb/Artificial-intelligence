from .base import PromptBuilder
from .modifiers import apply_prompt_modifiers

class GPTOSSPromptBuilder(PromptBuilder):
    def build_prompt(self, model_id, docs, query, chat_history=None):
        # System message describing the assistant's behavior
        system = (
            "You are a helpful assistant. ONLY use the provided documents to summarize "
            "the user's requested release. Provide a bullet-point list of the top 10 "
            "most important changes. If the answer is not in the documents, say so."
        )

        # Build the base prompt
        prompt_parts = [
            f"System:\n{system}\n",
        ]

        # Include any previous conversation (if provided)
        if chat_history:
            for turn in chat_history:
                role = turn["role"].capitalize()
                content = turn["content"]
                prompt_parts.append(f"{role}:\n{content}\n")

        # Add the new user query
        prompt_parts.append(
            "User:\n"
            f"=== DOCUMENTS ===\n{docs}\n\n"
            f"=== QUESTION ===\n{query}\n"
        )

        prompt_parts.append("Assistant:\n")

        prompt = "\n".join(prompt_parts)

        return apply_prompt_modifiers(prompt, query)
