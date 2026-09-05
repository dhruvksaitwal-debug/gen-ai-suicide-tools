import unittest

from query_expander import QueryExpander


class _FakeLLMClient:
    def __init__(self, response: str):
        self.response = response

    def chat_completion(self, messages, **kwargs):
        return self.response


class TestQueryExpander(unittest.TestCase):
    def test_strips_numbered_list_markers(self):
        expander = QueryExpander(_FakeLLMClient("1. First variation\n2) Second variation\n3. Third variation"))
        result = expander.expand("original query")

        self.assertEqual(
            result,
            ["original query", "First variation", "Second variation", "Third variation"],
        )

    def test_strips_bullet_markers_and_quotes(self):
        expander = QueryExpander(_FakeLLMClient('- "Bulleted variation"\n* Another one'))
        result = expander.expand("original query")

        self.assertEqual(result, ["original query", "Bulleted variation", "Another one"])

    def test_drops_blank_lines(self):
        expander = QueryExpander(_FakeLLMClient("First variation\n\n\nSecond variation"))
        result = expander.expand("original query")

        self.assertEqual(result, ["original query", "First variation", "Second variation"])


if __name__ == "__main__":
    unittest.main()
