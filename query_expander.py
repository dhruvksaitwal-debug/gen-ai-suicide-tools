import re


class QueryExpander:
    """Generates LLM-rephrased variations of a query to broaden retrieval recall."""

    # Strips leading list markers ("1.", "1)", "-", "*", "•") the LLM may prepend to each
    # variation, plus wrapping quotes, so raw list formatting doesn't get embedded as if it
    # were query text.
    _PREFIX_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def expand(self, query: str, num_variations: int = 3) -> list[str]:
        """Return the original query plus up to `num_variations` LLM-generated rephrasings."""
        prompt = f"Generate {num_variations} variations of: {query}"
        response = self.llm_client.chat_completion([
            {"role": "system", "content": "You rephrase queries."},
            {"role": "user", "content": prompt}
        ])

        variations = []
        for line in response.split("\n"):
            cleaned = self._PREFIX_RE.sub("", line).strip().strip("\"'")
            if cleaned:
                variations.append(cleaned)
        return [query] + variations
