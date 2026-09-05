import hashlib
import logging
import re
import threading

import chromadb
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.documents import Document

from cache import content_fingerprint

logger = logging.getLogger(__name__)

# Chroma's PersistentClient is not safe to instantiate separately, multiple times, against
# the same on-disk directory from concurrent threads (main.py processes several PDFs
# concurrently, each building its own orchestrator/VectorStoreManager) — confirmed
# empirically: distinct client instances race on reading each other's freshly-written HNSW
# segments, intermittently raising "Error creating hnsw segment reader: Nothing found on
# disk" for whichever instance loses the race. Every caller pointed at the same db_path
# shares one client instance instead.
_client_cache: dict[str, "chromadb.ClientAPI"] = {}
_client_cache_lock = threading.Lock()

# Sharing one client instance (get_shared_chroma_client) stops separate clients from racing
# on each other's on-disk state, but concurrent *calls* into that shared client from multiple
# threads still aren't fully safe: collection creation/deletion racing a query on a different
# collection has been observed to intermittently raise both "NotFoundError: Collection ...
# does not exist" and "HNSW segment reader: nothing found on disk", even against a single
# shared client. Serializing every call into the client with one process-wide lock (also used
# by orchestrator.py around vectorstore queries) eliminates the race; the actual expensive
# work — embedding and LLM calls — happens outside this lock, so it isn't the bottleneck.
chroma_lock = threading.Lock()


def get_shared_chroma_client(db_path: str) -> "chromadb.ClientAPI":
    with _client_cache_lock:
        client = _client_cache.get(db_path)
        if client is None:
            client = chromadb.PersistentClient(path=db_path)
            _client_cache[db_path] = client
        return client


def drop_collections(client: "chromadb.ClientAPI", names: list[str]) -> None:
    """
    Force-delete the named collections, ignoring any that don't exist. Used to give a failed
    document's retry attempt a guaranteed-clean slate: create_collection()'s own
    fingerprint-mismatch delete+recreate is meant to self-heal a stale collection, but a
    collection left behind by a crashed attempt can be registered in Chroma's catalog with
    incomplete on-disk segment data, which that same-attempt repair logic can't distinguish
    from a valid one — a query against it then fails with "Collection ... does not exist"
    even though get_collection() just found it. Dropping first removes that ambiguity.
    """
    with chroma_lock:
        for name in names:
            try:
                client.delete_collection(name)
            except NotFoundError:
                pass


class VectorStoreManager:
    """Creates Chroma collections backed by a persistent on-disk client."""

    def __init__(self, embedding_model, db_path: str = "./db"):
        self.client = get_shared_chroma_client(db_path)
        self.embedding_model = embedding_model
        self.db_path = db_path

    def create_collection(self, name: str, documents: list[Document]) -> Chroma:
        """
        Create (or reuse) a collection with the given name populated with `documents`.
        If a collection with this name already holds the same document content
        (by fingerprint), it's reused as-is rather than re-embedding everything from
        scratch. Otherwise any pre-existing collection of the same name is dropped
        first, so reruns on changed content don't accumulate duplicate embeddings.
        """
        fingerprint = content_fingerprint(*(d.page_content for d in documents))

        with chroma_lock:
            try:
                existing = self.client.get_collection(name)
                if existing.metadata and existing.metadata.get("fingerprint") == fingerprint:
                    logger.info("Reusing cached collection '%s' (content unchanged).", name)
                    return Chroma(
                        collection_name=name,
                        embedding_function=self.embedding_model,
                        client=self.client,
                        persist_directory=self.db_path
                    )
            except NotFoundError:
                pass

            try:
                self.client.delete_collection(name)
                logger.info("Replaced existing collection '%s'.", name)
            except NotFoundError:
                pass

            store = Chroma(
                collection_name=name,
                embedding_function=self.embedding_model,
                client=self.client,
                persist_directory=self.db_path,
                collection_metadata={"fingerprint": fingerprint}
            )
            # Chroma's add_documents rejects an empty batch outright; an empty collection is a
            # valid state (e.g. a short document with no chunks long enough for hypothetical-
            # question generation) and queries against it just return no matches.
            if documents:
                store.add_documents(documents=documents)
            return store
    
    @staticmethod
    def sanitize_collection_name(name: str) -> str:
        # Lowercase
        name = name.lower()

        # Replace spaces with underscores
        name = name.replace(" ", "_")

        # Remove invalid characters (keep only a-z, 0-9, ., _, -)
        name = re.sub(r"[^a-z0-9._-]", "", name)

        # Ensure it starts with alphanumeric
        if not re.match(r"^[a-z0-9]", name):
            name = "c_" + name

        # Ensure it ends with alphanumeric
        if not re.match(r".*[a-z0-9]$", name):
            name = name + "0"

        # Enforce max length (Chroma allows up to 512). Plain truncation collides whenever two
        # names share a long common prefix — this happened for real: a doc_id long enough that
        # both its "_article" and "_hypo" collection names truncated to the same 200 characters,
        # so creating the second collection deleted-and-replaced the first one out from under an
        # already-built vectorstore reference (same class of bug naming.py's safe_doc_id_stem
        # already fixes for output filenames). Only names that actually need shortening get a
        # content-hash suffix, which is what guarantees the result stays unique.
        maxlen = 200
        if len(name) <= maxlen:
            return name
        suffix = "_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        prefix_len = max(maxlen - len(suffix), 1)
        return name[:prefix_len] + suffix