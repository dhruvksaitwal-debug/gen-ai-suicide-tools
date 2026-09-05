"""
Disk cache for expensive per-PDF LLM outputs (extraction, hypothetical questions).

Each cache entry is keyed by (doc_id, kind) and tagged with a fingerprint of whatever
content actually determines its validity. A cache hit requires the stored fingerprint to
match the current one, so callers don't need to manage invalidation manually.
"""
import hashlib
import json
import logging
import os
from typing import Any, Callable

from constants import CACHE_DIR

logger = logging.getLogger(__name__)


def _hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def file_fingerprint(path: str) -> str:
    """Fingerprint based on file size + mtime; changes whenever the file's contents change."""
    stat = os.stat(path)
    return _hash(f"{stat.st_size}-{stat.st_mtime}")


def content_fingerprint(*parts: str) -> str:
    """Fingerprint based on the actual text content driving a computation."""
    return _hash("\x1f".join(parts))


def _cache_path(doc_id: str, kind: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{doc_id}__{kind}.json")


def load_or_compute(doc_id: str, kind: str, fingerprint: str, compute: Callable[[], Any], force: bool = False) -> Any:
    """
    Return cached JSON-serializable data for (doc_id, kind) if it exists and its stored
    fingerprint matches `fingerprint`; otherwise compute it via `compute()` and cache the result.
    """
    path = _cache_path(doc_id, kind)

    if not force and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("fingerprint") == fingerprint:
                logger.info("Using cached %s for %s", kind, doc_id)
                return cached["data"]
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning("Ignoring unreadable cache file %s: %s", path, e)

    data = compute()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint, "data": data}, f)
    return data
