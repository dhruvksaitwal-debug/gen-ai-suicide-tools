import logging
import time
import random
from typing import Callable

from openai import APIConnectionError
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingsWithBackoff(OpenAIEmbeddings):
    """OpenAIEmbeddings subclass that retries on rate limits with exponential backoff."""

    def _with_backoff(self, fn: Callable, *args, **kwargs):
        delay = 1
        max_delay = 60
        max_retries = 12

        for attempt in range(max_retries):
            try:
                return fn(*args, **kwargs)

            except Exception as e:
                msg = str(e).lower()
                # Connection errors (DNS blips, resets, timeouts) are just as transient as
                # rate limits and deserve the same backoff-and-retry treatment (see
                # llm_client.py's identical handling).
                if "rate limit" in msg or "429" in msg or isinstance(e, APIConnectionError):
                    logger.warning(
                        "[embeddings] Rate limit or connection error hit (attempt %d/%d): %s. Sleeping %.1fs...",
                        attempt + 1, max_retries, e, delay
                    )
                    time.sleep(delay + random.uniform(0, 0.5))
                    delay = min(delay * 2, max_delay)
                    continue

                raise

        raise RuntimeError("Exceeded maximum retries for embeddings due to rate limits or connection errors.")

    def embed_query(self, text: str) -> list[float]:
        return self._with_backoff(super().embed_query, text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._with_backoff(super().embed_documents, texts)