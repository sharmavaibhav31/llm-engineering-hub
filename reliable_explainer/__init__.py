"""Reliable Explainer: Production-grade streaming and resilient LLM client."""

from reliable_explainer.batch import BatchItemResult, run_batch
from reliable_explainer.config import ExplainerConfig, load_config
from reliable_explainer.explainer import StreamResult, explain_stream
from reliable_explainer.format import (
    Result,
    format_answer,
    format_question,
    format_retry_notice,
    format_summary_table,
)
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
    "run_batch",
    "BatchItemResult",
    "Result",
    "format_question",
    "format_retry_notice",
    "format_answer",
    "format_summary_table",
]
