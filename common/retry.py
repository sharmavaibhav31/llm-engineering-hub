"""Production-grade retry utility with exponential backoff and jitter.

Differentiates retryable vs non-retryable errors, respects Retry-After headers,
and uses non-blocking asyncio.sleep to prevent thread starvation.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, Optional, TypeVar
import httpx
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

logger = logging.getLogger("reliable_explainer.retry")

T = TypeVar("T")

# Errors that should never be retried because they represent client-side bugs or invalid requests
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}
NON_RETRYABLE_EXCEPTIONS = (
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    UnprocessableEntityError,
)


def is_retryable_error(exc: Exception) -> bool:
    """Classify whether an exception is safe and worthwhile to retry.

    Retryable:
        - 429 RateLimitError (Groq rate limits)
        - 5xx Server Errors (500, 502, 503, 504)
        - APITimeoutError / httpx.TimeoutException (network/generation timeouts)
        - APIConnectionError / httpx.NetworkError (socket/connection drops)

    Non-retryable:
        - 400 Bad Request (bad prompt, invalid parameters)
        - 401 Authentication (bad API key)
        - 403 Forbidden
        - 404 Model Not Found
        - 422 Unprocessable Entity
    """
    if isinstance(exc, NON_RETRYABLE_EXCEPTIONS):
        return False

    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True

    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True

    if isinstance(exc, APIStatusError):
        status = exc.status_code
        if status in NON_RETRYABLE_STATUS_CODES:
            return False
        if status >= 500:
            return True
        if status == 429:
            return True

    return False


def extract_retry_after(exc: Exception) -> Optional[float]:
    """Extract 'Retry-After' header duration in seconds if present in response."""
    if isinstance(exc, APIStatusError) and exc.response is not None:
        header_val = exc.response.headers.get("retry-after")
        if header_val:
            try:
                # Groq / standard servers send seconds as float or integer
                return max(0.0, float(header_val))
            except ValueError:
                pass
    return None


def calculate_backoff(
    attempt: int,
    base_backoff: float = 0.5,
    max_backoff: float = 8.0,
    retry_after: Optional[float] = None,
) -> float:
    """Calculate sleep time using Full Jitter exponential backoff.

    Formula:
        If Retry-After is present: max(retry_after, random(0, 0.5)) to break lockstep.
        Otherwise (Full Jitter): uniform_random(0, min(max_backoff, base * 2^attempt))

    Full Jitter completely disperses thundering herds of colliding requests
    re-attempting at the same millisecond.
    """
    if retry_after is not None:
        # Respect server header, add slight jitter (up to 200ms) to avoid thundering herd on server reset
        return min(max_backoff, retry_after + random.uniform(0.05, 0.2))

    # Full Jitter backoff
    exponential_cap = min(max_backoff, base_backoff * (2.0 ** attempt))
    return random.uniform(0.0, exponential_cap)


async def execute_with_retry(
    operation: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_backoff: float = 0.5,
    max_backoff: float = 8.0,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> T:
    """Execute an async operation with exponential backoff and jitter.

    Args:
        operation: Async callable returning a value.
        max_retries: Max retry attempts (0 means try once, no retries).
        base_backoff: Base delay in seconds.
        max_backoff: Delay ceiling in seconds.
        on_retry: Optional hook (attempt, exception, delay) for telemetry / logging.

    Returns:
        The result of the async operation.

    Raises:
        The original exception if non-retryable or if max_retries is exhausted.
    """
    attempt = 0
    while True:
        try:
            return await operation()
        except Exception as exc:
            if attempt >= max_retries or not is_retryable_error(exc):
                # Either we reached limit or the error cannot be retried
                logger.error(
                    f"Operation failed permanently (attempt {attempt + 1}/{max_retries + 1}): "
                    f"{type(exc).__name__}: {exc}"
                )
                raise exc

            retry_after = extract_retry_after(exc)
            delay = calculate_backoff(
                attempt=attempt,
                base_backoff=base_backoff,
                max_backoff=max_backoff,
                retry_after=retry_after,
            )

            logger.warning(
                f"Retryable error {type(exc).__name__} (status={getattr(exc, 'status_code', 'N/A')}). "
                f"Attempt {attempt + 1}/{max_retries}. Backing off for {delay:.2f}s..."
            )

            if on_retry:
                on_retry(attempt + 1, exc, delay)

            # Non-blocking async sleep — preserves concurrent tasks on event loop
            await asyncio.sleep(delay)
            attempt += 1

