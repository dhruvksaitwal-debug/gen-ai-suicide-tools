class RAGAnswerer:
    def __init__(self, llm_client, system_prompt, user_prompt_template):
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    def answer(self, query, contexts):
        context_str = "\n---\n".join(contexts)
        messages = [
            {"role": "developer", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_template.format(context=context_str, question=query)}
        ]
        return self.llm_client.chat_completion(messages)

    def combine(self, ans1, ans2):
        messages = [
            {"role": "system", "content": "Combine two answers into one cohesive response."},
            {"role": "user", "content": f"Answer 1: {ans1}\nAnswer 2: {ans2}"}
        ]
        return self.llm_client.chat_completion(messages)