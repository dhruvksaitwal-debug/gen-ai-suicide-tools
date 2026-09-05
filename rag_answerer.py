class RAGAnswerer:
    """Answers a query against a set of retrieved contexts."""

    def __init__(self, llm_client, system_prompt: str, user_prompt_template: str):
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    def answer(self, query: str, contexts: list[str]) -> str:
        context_str = "\n---\n".join(contexts)
        messages = [
            {"role": "developer", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_template.format(context=context_str, question=query)}
        ]
        return self.llm_client.chat_completion(messages)