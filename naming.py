"""Filesystem-safe naming for doc_id-derived output paths (result CSVs, image folders)."""
import hashlib
import re

from constants import PDF_STEM_MAXLEN

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_doc_id_stem(doc_id: str, maxlen: int = PDF_STEM_MAXLEN) -> str:
    """
    Shorten `doc_id` to a filesystem-safe stem of at most `maxlen` characters, for use in
    output filenames and folder names.

    Plain truncation collides silently whenever two doc_ids share a long common prefix —
    e.g. two real articles in this corpus, "Development and Validation of a Nomogram for
    Predicting Suicidal Ideation Among Rural Adolescents in China" and "Development and
    validation of the Durham Risk Score for estimating suicide attempt risk...", both
    truncate to "development_and_validation_of_" and would overwrite each other's results.
    Short doc_ids that already fit within `maxlen` are returned unchanged (no suffix
    needed, nothing lost by not truncating); only doc_ids that must be shortened get a
    short content-hash suffix, which is what actually guarantees the result stays unique.
    """
    cleaned = _INVALID_CHARS.sub("_", doc_id).strip().lower().replace(" ", "_")
    if len(cleaned) <= maxlen:
        return cleaned

    suffix = "_" + hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:8]
    prefix_len = max(maxlen - len(suffix), 1)
    return cleaned[:prefix_len] + suffix
