import unittest
from unittest import mock

import pandas as pd

from pdf_extractor import PDFExtractor, _rmtree_with_retry


class _FakeLLMClient:
    def __init__(self, fail_on_substring=None):
        self.fail_on_substring = fail_on_substring

    def chat_completion(self, messages, **kwargs):
        content = messages[-1]["content"]
        if self.fail_on_substring and self.fail_on_substring in content:
            raise RuntimeError("simulated permanent API failure")
        return "a table summary"

    def describe_image(self, image_bytes, **kwargs):
        return "an image description"


class TestPDFExtractorTableResilience(unittest.TestCase):
    def test_successful_table_produces_summary_entry(self):
        extractor = PDFExtractor(_FakeLLMClient())
        df = pd.DataFrame({"a": [1, 2]})

        result = extractor._table_entry(df, page_num=0, table_index=0)

        self.assertIn("Table 1 Summary", result)
        self.assertIn("a table summary", result)

    def test_failing_table_summarization_degrades_gracefully(self):
        # A failed LLM call for one table must not raise out of _table_entry: it's run as a
        # thread-pool future in extract(), and an unhandled exception there would crash
        # extraction for the whole PDF (losing every other page/table/image) over one table.
        extractor = PDFExtractor(_FakeLLMClient(fail_on_substring="poison"))
        df = pd.DataFrame({"poison": [1]})

        result = extractor._table_entry(df, page_num=0, table_index=0)

        self.assertIn("Table 1 Error", result)
        self.assertIn("Could not process table", result)


class TestRmtreeWithRetry(unittest.TestCase):
    # Cloud-sync clients (OneDrive, Dropbox) commonly hold a brief lock on a just-created
    # or just-modified directory, raising a transient PermissionError if it's deleted too
    # soon after. This must retry through that instead of crashing the whole extraction.
    def test_recovers_from_transient_permission_error(self):
        calls = []

        def flaky_rmtree(path):
            calls.append(path)
            if len(calls) < 3:
                raise PermissionError("[WinError 5] Access is denied")

        with mock.patch("shutil.rmtree", side_effect=flaky_rmtree):
            _rmtree_with_retry("some/path", attempts=5, delay=0.01)

        self.assertEqual(len(calls), 3)

    def test_raises_after_exhausting_retries(self):
        def always_fails(path):
            raise PermissionError("persistent lock")

        with mock.patch("shutil.rmtree", side_effect=always_fails):
            with self.assertRaises(PermissionError):
                _rmtree_with_retry("some/path", attempts=3, delay=0.01)


if __name__ == "__main__":
    unittest.main()
