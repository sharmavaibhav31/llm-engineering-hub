"""Configuration and settings for LCEL Pipeline."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


@dataclass(frozen=True)
class PipelineSettings:
    """Resolved configuration settings for LCEL Pipeline."""
    groq_api_key: str
    model: str
    chunk_size: int
    chunk_overlap: int


def validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    """Validate that chunk overlap does not exceed chunk size.

    Exits with a friendly error message instead of raising a traceback.
    """
    if chunk_overlap > chunk_size:
        sys.stderr.write(
            f"Error: chunk_overlap ({chunk_overlap}) cannot be larger than chunk_size ({chunk_size}).\n"
            f"Please reduce --chunk-overlap or increase --chunk-size.\n"
        )
        sys.exit(1)


def load_settings(
    model_override: Optional[str] = None,
    chunk_size_override: Optional[int] = None,
    chunk_overlap_override: Optional[int] = None,
    require_api_key: bool = True,
) -> PipelineSettings:
    """Load and validate pipeline settings from environment and CLI flags.

    Args:
        model_override: Optional model name from CLI flag.
        chunk_size_override: Optional chunk size from CLI flag.
        chunk_overlap_override: Optional chunk overlap from CLI flag.
        require_api_key: Whether to enforce GROQ_API_KEY presence.

    Returns:
        PipelineSettings dataclass.

    Exits:
        Exits with code 1 if GROQ_API_KEY is missing (when required)
        or if chunk_overlap > chunk_size.
    """
    load_dotenv()

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if require_api_key and not api_key:
        sys.stderr.write(
            "Error: GROQ_API_KEY environment variable is missing or empty.\n"
            "Please set it in your environment or .env file before running.\n"
        )
        sys.exit(1)

    model = (
        model_override
        or os.environ.get("EXPLAINER_MODEL", "").strip()
        or os.environ.get("DEFAULT_MODEL", "").strip()
        or DEFAULT_MODEL
    )

    chunk_size = chunk_size_override if chunk_size_override is not None else DEFAULT_CHUNK_SIZE
    chunk_overlap = chunk_overlap_override if chunk_overlap_override is not None else DEFAULT_CHUNK_OVERLAP

    validate_chunk_settings(chunk_size, chunk_overlap)

    return PipelineSettings(
        groq_api_key=api_key,
        model=model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

