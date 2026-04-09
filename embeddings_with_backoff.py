import time
import random
from langchain_openai import OpenAIEmbeddings

class EmbeddingsWithBackoff(OpenAIEmbeddings):

    def _with_backoff(self, fn, *args, **kwargs):
        delay = 1
        max_delay = 60
        max_retries = 12

        for attempt in range(max_retries):
            try:
                return fn(*args, **kwargs)

            except Exception as e:
                msg = str(e).lower()
                if "rate limit" in msg or "429" in msg:
                    print(f"[Backoff: embeddings] Sleeping {delay:.1f} seconds...")
                    time.sleep(delay + random.uniform(0, 0.5))
                    delay = min(delay * 2, max_delay)
                    continue

                raise

        raise RuntimeError("Exceeded maximum retries for embeddings.")

    def embed_query(self, text):
        return self._with_backoff(super().embed_query, text)

    def embed_documents(self, texts):
        return self._with_backoff(super().embed_documents, texts)