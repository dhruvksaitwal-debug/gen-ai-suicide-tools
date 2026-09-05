"""Shared environment/config loading for entry points that talk to OpenAI."""
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


class MissingEnvVarError(RuntimeError):
    """Raised when a required environment variable is not set."""


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    base_url: Optional[str]


def load_openai_config() -> OpenAIConfig:
    """Load and validate OpenAI credentials from the environment/.env file."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise MissingEnvVarError(
            "OPENAI_API_KEY is not set. Add it to your .env file or environment "
            "(see .env.example)."
        )
    return OpenAIConfig(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))
