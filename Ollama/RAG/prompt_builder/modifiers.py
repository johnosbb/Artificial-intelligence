def apply_prompt_modifiers(prompt, query):
    q = query.lower()

    if "summarize" in q or "summarise" in q:
        prompt = "[TASK:SUMMARIZE]\n" + prompt
    elif "show me" in q:
        prompt = "[TASK:SHOW]\n" + prompt
    elif "list" in q and "steps" in q:
        prompt = "[TASK:LIST_STEPS]\n" + prompt

    return prompt
