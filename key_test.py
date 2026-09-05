"""Manual smoke test: confirms OPENAI_API_KEY/OPENAI_BASE_URL are valid."""
import logging

from openai import OpenAI

from config import load_openai_config

logger = logging.getLogger(__name__)


def main():
    config = load_openai_config()
    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="hello world"
    )
    logger.info("OK: received embedding of length %d", len(response.data[0].embedding))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()