import unittest

from naming import safe_doc_id_stem


class TestSafeDocIdStem(unittest.TestCase):
    def test_short_doc_id_is_unchanged_besides_lowercasing(self):
        self.assertEqual(safe_doc_id_stem("Gold01"), "gold01")

    def test_long_titles_that_share_a_common_prefix_do_not_collide(self):
        # Real pair from the corpus: identical for the first 32 characters.
        a = (
            "Development and Validation of a Nomogram for Predicting Suicidal "
            "Ideation Among Rural Adolescents in China_compressed"
        )
        b = (
            "Development and validation of the Durham Risk Score for estimating "
            "suicide attempt risk- A prospective cohort analysis_compressed"
        )
        stem_a, stem_b = safe_doc_id_stem(a), safe_doc_id_stem(b)
        self.assertNotEqual(stem_a, stem_b)
        self.assertLessEqual(len(stem_a), 30)
        self.assertLessEqual(len(stem_b), 30)

    def test_same_doc_id_always_produces_the_same_stem(self):
        long_id = "A" * 100
        self.assertEqual(safe_doc_id_stem(long_id), safe_doc_id_stem(long_id))

    def test_result_respects_custom_maxlen(self):
        stem = safe_doc_id_stem("A" * 100, maxlen=20)
        self.assertLessEqual(len(stem), 20)

    def test_invalid_filename_characters_are_replaced(self):
        stem = safe_doc_id_stem('Title: A "Study" of X/Y <2020>')
        for ch in '<>:"/\\|?*':
            self.assertNotIn(ch, stem)

    def test_spaces_become_underscores_and_result_is_lowercase(self):
        stem = safe_doc_id_stem("Short Title Here")
        self.assertEqual(stem, "short_title_here")


if __name__ == "__main__":
    unittest.main()
