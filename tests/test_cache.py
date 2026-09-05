import shutil
import tempfile
import unittest

import cache


class TestCache(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        self._orig_cache_dir = cache.CACHE_DIR
        cache.CACHE_DIR = self._tmp_dir

    def tearDown(self):
        cache.CACHE_DIR = self._orig_cache_dir
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_cache_hit_avoids_recompute(self):
        calls = []

        def compute():
            calls.append(1)
            return {"result": "expensive"}

        first = cache.load_or_compute("doc1", "extraction", "fingerprint-a", compute)
        second = cache.load_or_compute("doc1", "extraction", "fingerprint-a", compute)

        self.assertEqual(first, {"result": "expensive"})
        self.assertEqual(second, {"result": "expensive"})
        self.assertEqual(len(calls), 1, "second call with the same fingerprint should hit the cache")

    def test_fingerprint_change_forces_recompute(self):
        calls = []

        def compute():
            calls.append(1)
            return {"value": len(calls)}

        first = cache.load_or_compute("doc1", "extraction", "fingerprint-a", compute)
        second = cache.load_or_compute("doc1", "extraction", "fingerprint-b", compute)

        self.assertNotEqual(first, second)
        self.assertEqual(len(calls), 2, "a changed fingerprint should invalidate the cache")

    def test_force_bypasses_cache(self):
        calls = []

        def compute():
            calls.append(1)
            return {"value": len(calls)}

        cache.load_or_compute("doc1", "extraction", "fingerprint-a", compute)
        cache.load_or_compute("doc1", "extraction", "fingerprint-a", compute, force=True)

        self.assertEqual(len(calls), 2, "force=True should always recompute")

    def test_corrupted_cache_file_falls_back_to_recompute(self):
        path = cache._cache_path("doc1", "extraction")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not valid json")

        result = cache.load_or_compute("doc1", "extraction", "fingerprint-a", lambda: {"ok": True})
        self.assertEqual(result, {"ok": True})

    def test_content_fingerprint_stable_and_sensitive_to_input(self):
        self.assertEqual(cache.content_fingerprint("a", "b"), cache.content_fingerprint("a", "b"))
        self.assertNotEqual(cache.content_fingerprint("a", "b"), cache.content_fingerprint("a", "c"))

    def test_file_fingerprint_changes_when_file_changes(self):
        import os
        import time

        path = f"{self._tmp_dir}/probe.txt"
        with open(path, "w") as f:
            f.write("v1")
        fp1 = cache.file_fingerprint(path)

        time.sleep(0.01)
        with open(path, "w") as f:
            f.write("v1 but longer")
        fp2 = cache.file_fingerprint(path)

        self.assertNotEqual(fp1, fp2)


if __name__ == "__main__":
    unittest.main()
