"""Reliable Explainer: Production-grade streaming and resilient LLM client."""

from reliable_explainer.config import ExplainerConfig, load_config
from reliable_explainer.explainer import StreamResult, explain_stream
from reliable_explainer.retry import (
    RetryResult,
    calculate_delay,
    is_retryable_error,
    with_retry,
)

__all__ = [
    "ExplainerConfig",
    "load_config",
    "StreamResult",
    "explain_stream",
    "with_retry",
    "RetryResult",
    "is_retryable_error",
    "calculate_delay",
]
