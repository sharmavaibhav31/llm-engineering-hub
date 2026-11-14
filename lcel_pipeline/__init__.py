"""LCEL Pipeline: LangChain Expression Language and Text Ingestion Mini-Project."""

from lcel_pipeline.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL,
    PipelineSettings,
    load_settings,
    validate_chunk_settings,
)
from lcel_pipeline.document import (
    SplitSummary,
    format_document_output,
    load_and_split_document,
    run_document_mode,
)
from lcel_pipeline.errors import (
    AUTH_ERROR_MSG,
    NETWORK_ERROR_MSG,
    RATE_LIMIT_ERROR_MSG,
    handle_cli_error,
)
from lcel_pipeline.explain import (
    build_explain_chain,
    build_explain_prompt,
    run_explain,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "PipelineSettings",
    "load_settings",
    "validate_chunk_settings",
    "build_explain_chain",
    "build_explain_prompt",
    "run_explain",
    "SplitSummary",
    "load_and_split_document",
    "run_document_mode",
    "format_document_output",
    "handle_cli_error",
    "AUTH_ERROR_MSG",
    "RATE_LIMIT_ERROR_MSG",
    "NETWORK_ERROR_MSG",
]
