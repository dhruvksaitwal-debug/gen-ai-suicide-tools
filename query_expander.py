class QueryExpander:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def expand(self, query, num_variations=3):
        prompt = f"Generate {num_variations} variations of: {query}"
        response = self.llm_client.chat_completion([
            {"role": "system", "content": "You rephrase queries."},
            {"role": "user", "content": prompt}
        ])
        return [query] + [v.strip() for v in response.split("\n") if v.strip()]