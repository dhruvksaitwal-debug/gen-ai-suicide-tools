import unittest

from langchain_core.documents import Document

from hypo_question_generator import HypotheticalQuestionGenerator


class _FakeLLMClient:
    def __init__(self, fail_on_substring=None):
        self.fail_on_substring = fail_on_substring

    def chat_completion(self, messages, **kwargs):
        content = messages[-1]["content"]
        if self.fail_on_substring and self.fail_on_substring in content:
            raise RuntimeError("simulated permanent API failure")
        return "Question one?\nQuestion two?"


class TestHypotheticalQuestionGenerator(unittest.TestCase):
    def test_generates_one_document_per_question_line(self):
        docs = [Document(id=1, page_content="chunk text")]
        generator = HypotheticalQuestionGenerator(_FakeLLMClient(), max_workers=2)

        questions = generator.generate(docs, "system message")

        self.assertEqual(len(questions), 2)
        # langchain's Document coerces `id` to str regardless of what's passed in, so the
        # parent_chunk_id propagated into metadata is "1", not the int 1 — this matters
        # because orchestrator.py's HyDE-context lookup relies on both sides matching.
        self.assertEqual(questions[0].metadata["parent_chunk_id"], "1")

    def test_one_failing_chunk_does_not_crash_the_whole_batch(self):
        docs = [Document(id=i, page_content=f"chunk {i}") for i in range(4)]
        generator = HypotheticalQuestionGenerator(_FakeLLMClient(fail_on_substring="chunk 2"), max_workers=4)

        questions = generator.generate(docs, "system message")

        # 3 successful chunks x 2 questions each; the failing chunk contributes nothing
        # but must not raise and abort the other chunks' results.
        self.assertEqual(len(questions), 6)
        failed_chunk_ids = {q.metadata["parent_chunk_id"] for q in questions}
        self.assertNotIn(2, failed_chunk_ids)


if __name__ == "__main__":
    unittest.main()
