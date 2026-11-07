
"""Configuration module for the Reliable Explainer CLI.

Single source of truth for environment variables (GROQ_API_KEY, EXPLAINER_MODEL).
No other module in reliable_explainer may read os.environ directly.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from groq import AsyncGroq

DEFAULT_MODEL = "openai/gpt-oss-20b"


@dataclass(frozen=True)
class ExplainerConfig:
    """Resolved runtime configuration for the Explainer application."""
    api_key: str
    model: str
    client: AsyncGroq


def load_config(model_override: Optional[str] = None) -> ExplainerConfig:
    """Load configuration from environment variables and CLI overrides.

    Args:
        model_override: Optional model name passed via CLI flag (--model).

    Returns:
        ExplainerConfig with validated API key, resolved model, and AsyncGroq client.

    Exits:
        Exits with code 1 and prints a single clear error message to stderr
        if GROQ_API_KEY is not set or empty.
    """
    # Load .env if present (does not overwrite existing environment variables)
    load_dotenv()

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        sys.stderr.write(
            "Error: GROQ_API_KEY environment variable is missing or empty.\n"
            "Please set it in your environment or .env file.\n"
        )
        sys.exit(1)

    # Resolution order: CLI flag override -> EXPLAINER_MODEL env var -> DEFAULT_MODEL
    model = (
        model_override
        or os.environ.get("EXPLAINER_MODEL", "").strip()
        or DEFAULT_MODEL
    )

    # CRITICAL: max_retries=0 ensures the SDK does not retry internally,
    # leaving retry loops and telemetry completely under our control.
    client = AsyncGroq(
        api_key=api_key,
        max_retries=0,
    )

    return ExplainerConfig(
        api_key=api_key,
        model=model,
        client=client,
    )

