import shutil
import tempfile
import unittest

from langchain_core.documents import Document

from vectorstore_manager import VectorStoreManager


class _FakeEmbeddings:
    """Deterministic, dependency-free stand-in for a real embedding model."""

    def embed_documents(self, texts):
        return [[float(len(t) % 7), 0.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7), 0.0]


class TestVectorStoreManagerIdempotency(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        self.manager = VectorStoreManager(_FakeEmbeddings(), db_path=self._tmp_dir)

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_rerun_with_same_content_does_not_duplicate(self):
        docs = [Document(id=i, page_content=f"chunk {i}") for i in range(3)]

        first = self.manager.create_collection("col", docs)
        second = self.manager.create_collection("col", docs)

        self.assertEqual(first._collection.count(), 3)
        self.assertEqual(second._collection.count(), 3, "same content rerun must not duplicate embeddings")

    def test_empty_document_list_does_not_crash(self):
        # e.g. a short document with no chunks long enough for hypothetical-question
        # generation; add_documents([]) raises in chromadb, so this must be special-cased.
        store = self.manager.create_collection("empty_col", [])
        self.assertEqual(store._collection.count(), 0)
        self.assertEqual(store.similarity_search_by_vector([1.0, 0.0], k=5), [])

    def test_empty_then_populated_transitions_correctly(self):
        self.manager.create_collection("col", [])
        docs = [Document(id=i, page_content=f"chunk {i}") for i in range(3)]
        populated = self.manager.create_collection("col", docs)
        self.assertEqual(populated._collection.count(), 3)

    def test_rerun_with_changed_content_replaces_collection(self):
        docs_v1 = [Document(id=i, page_content=f"chunk {i}") for i in range(3)]
        docs_v2 = [Document(id=i, page_content=f"different chunk {i}") for i in range(5)]

        self.manager.create_collection("col", docs_v1)
        replaced = self.manager.create_collection("col", docs_v2)

        self.assertEqual(replaced._collection.count(), 5)


class TestSanitizeCollectionName(unittest.TestCase):
    def test_lowercases_and_replaces_spaces(self):
        self.assertEqual(VectorStoreManager.sanitize_collection_name("My Doc Name"), "my_doc_name")

    def test_strips_invalid_characters(self):
        self.assertEqual(VectorStoreManager.sanitize_collection_name("doc:name(1).pdf"), "docname1.pdf")

    def test_forces_alphanumeric_start_and_end(self):
        result = VectorStoreManager.sanitize_collection_name("_leading_and_trailing_")
        self.assertRegex(result, r"^[a-z0-9]")
        self.assertRegex(result, r"[a-z0-9]$")

    def test_truncates_to_max_length(self):
        result = VectorStoreManager.sanitize_collection_name("a" * 500)
        self.assertLessEqual(len(result), 200)

    def test_long_names_sharing_a_prefix_do_not_collide(self):
        # Real pair from the corpus: a >200-char doc_id whose "_article" and "_hypo" variants
        # both truncated to the identical 200 characters, so building the second collection
        # deleted-and-replaced the first one out from under an already-built vectorstore
        # reference — every query against it then failed with "Collection ... does not exist".
        base = (
            "Reliability, validity and factorial structure of the Arabic version of the "
            "international suicide prevention trial (InterSePT) scale for suicidal thinking "
            "in schizophrenia patients in Doha, Qatar [PMC5142345]"
        )
        article = VectorStoreManager.sanitize_collection_name(base + "_article")
        hypo = VectorStoreManager.sanitize_collection_name(base + "_hypo")
        self.assertNotEqual(article, hypo)
        self.assertLessEqual(len(article), 200)
        self.assertLessEqual(len(hypo), 200)


if __name__ == "__main__":
    unittest.main()
