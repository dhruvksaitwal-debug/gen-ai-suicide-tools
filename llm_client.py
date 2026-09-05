import base64
import logging
import threading
import time
import random
from typing import Callable, Optional

from openai import APIConnectionError, OpenAI

from constants import LLM_CONCURRENCY_LIMIT

logger = logging.getLogger(__name__)

# Process-wide, shared across every LLMClient instance (one per document): the constraint
# being protected against is the OpenAI account's shared TPM ceiling, not any one instance's
# own throughput, so the cap has to apply globally rather than per-client.
_concurrency_semaphore = threading.Semaphore(LLM_CONCURRENCY_LIMIT)


class LLMClient:
    """Thin wrapper around the OpenAI chat API with rate-limit backoff."""

    def __init__(self, api_key: str, base_url: Optional[str], model_name: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def _with_backoff(self, fn: Callable, *args, **kwargs):
        delay = 1
        max_delay = 60
        max_retries = 8

        for attempt in range(max_retries):
            try:
                # Held only for the in-flight request itself, not across backoff sleeps —
                # otherwise a sleeping call would occupy a concurrency slot doing nothing.
                with _concurrency_semaphore:
                    return fn(*args, **kwargs)

            except Exception as e:
                msg = str(e).lower()
                # Connection errors (DNS blips, resets, timeouts) surface as openai's own
                # APIConnectionError, not a rate-limit message — they're just as transient and
                # deserve the same backoff-and-retry treatment rather than failing the
                # document outright on what's often a one-off network hiccup.
                if "rate limit" in msg or "429" in msg or isinstance(e, APIConnectionError):
                    logger.warning("Rate limit or connection error hit (attempt %d/%d): %s. Sleeping %.1fs...", attempt + 1, max_retries, e, delay)
                    time.sleep(delay + random.uniform(0, 0.5))
                    delay = min(delay * 2, max_delay)
                    continue

                raise

        raise RuntimeError("Exceeded maximum retries due to repeated rate limits or connection errors.")

    def chat_completion(self, messages: list[dict], temperature: float = 0, max_tokens: int = 500) -> str:
        """Send a chat completion request, retrying on rate limits."""
        response = self._with_backoff(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    

    def describe_image(self, image_bytes: bytes, max_tokens: int = 300) -> str:
        """Send image bytes to the LLM for description, retrying on rate limits."""
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail so that the output is less than 200 words."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ],
            }
        ]
        response = self._with_backoff(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()