"""Application configuration using Pydantic Settings.

Loads environment variables from .env file or system environment with typed
defaults for timeouts, retries, models, and concurrency limits.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from groq import AsyncGroq


class Settings(BaseSettings):
    """Central configuration for AI engineering services."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: Optional[str] = Field(
        default=None,
        description="Groq Cloud API Key (starts with gsk_)",
    )
    default_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Default LLM model identifier for Groq API",
    )
    request_timeout_seconds: float = Field(
        default=30.0,
        description="HTTP request timeout for LLM client calls",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts on retryable errors",
    )
    base_backoff_seconds: float = Field(
        default=0.5,
        description="Initial backoff delay in seconds for exponential backoff",
    )
    max_backoff_seconds: float = Field(
        default=8.0,
        description="Maximum backoff delay ceiling in seconds",
    )
    max_concurrency: int = Field(
        default=5,
        description="Default bounded concurrency limit for batch requests",
    )

    def get_async_groq_client(self) -> AsyncGroq:
        """Create an AsyncGroq client with SDK-level retries disabled.

        Disabling max_retries at the SDK level (max_retries=0) is crucial
        so our custom retry wrapper controls backoff, jitter, and telemetry.
        """
        return AsyncGroq(
            api_key=self.groq_api_key,
            timeout=self.request_timeout_seconds,
            max_retries=0,  # CRITICAL: Disable internal SDK retries to prevent retry multiplication
        )


# Singleton instance for simple imports across modules
settings = Settings()

