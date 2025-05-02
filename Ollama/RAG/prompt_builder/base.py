class PromptBuilder:
    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt or "You are a helpful assistant."

    def build_prompt(self, model_id, docs, query, chat_history=None):
        """
        Construct a prompt tailored for the given model.

        :param model_id: Identifier for the model (e.g., internlm2-chat).
        :param docs: Retrieved documents (string).
        :param query: User's question (string).
        :param chat_history: Optional chat history (list of dicts).
        """
        raise NotImplementedError("Subclasses must implement build_prompt.")
