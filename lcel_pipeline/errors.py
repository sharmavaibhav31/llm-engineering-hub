"""CLI Error boundary for LCEL Pipeline.

Translates provider exceptions into distinct, human-readable terminal messages
at the application boundary, keeping the LCEL chain pure.
"""

from __future__ import annotations

import sys
import httpx
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

AUTH_ERROR_MSG = "Error: Authentication failed. Please check that your GROQ_API_KEY is valid."
RATE_LIMIT_ERROR_MSG = "Error: Rate limit exceeded. Please wait a moment before trying again."
NETWORK_ERROR_MSG = "Error: Network connection failed. Please check your internet connection."


def handle_cli_error(exc: Exception) -> int:
    """Map exceptions to distinct, user-friendly CLI messages.

    Returns:
        Exit code (1).
    """
    if isinstance(exc, AuthenticationError) or (isinstance(exc, APIStatusError) and exc.status_code == 401):
        sys.stderr.write(f"{AUTH_ERROR_MSG}\n")
        return 1

    if isinstance(exc, RateLimitError) or (isinstance(exc, APIStatusError) and exc.status_code == 429):
        sys.stderr.write(f"{RATE_LIMIT_ERROR_MSG}\n")
        return 1

    if isinstance(exc, (APIConnectionError, APITimeoutError, httpx.NetworkError, httpx.TimeoutException, ConnectionError, TimeoutError)):
        sys.stderr.write(f"{NETWORK_ERROR_MSG}\n")
        return 1

    if isinstance(exc, APIStatusError):
        msg = getattr(exc, "message", str(exc))
        sys.stderr.write(f"Error: API service returned error ({exc.status_code}): {msg}\n")
        return 1

    sys.stderr.write(f"Error: {type(exc).__name__}: {exc}\n")
    return 1

