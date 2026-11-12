"""Async retry helper with exponential backoff, jitter, and error classification.

Retries only on transient errors (429 rate limit, 5xx server errors, timeouts, connection drops).
Never retries on client/auth errors (400, 401, 403, 404).
Honors Retry-After response headers and prevents retry multiplication.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, Optional, TypeVar

import httpx
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

logger = logging.getLogger("reliable_explainer.retry")

T = TypeVar("T")

# HTTP status codes and exceptions that must NEVER be retried
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}
NON_RETRYABLE_EXCEPTIONS = (
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    UnprocessableEntityError,
)


def is_retryable_error(exc: Exception) -> bool:
    """Determine whether an error is transient and safe to retry.

    Retryable:
        - 429 RateLimitError
        - 5xx Server Errors (500, 502, 503, 504)
        - APITimeoutError, httpx.TimeoutException, asyncio.TimeoutError
        - APIConnectionError, httpx.NetworkError, httpx.RemoteProtocolError, ConnectionError

    Non-retryable:
        - 400 Bad Request
        - 401 Authentication Error
        - 403 Forbidden
        - 404 Not Found
        - 422 Unprocessable Entity
    """
    if isinstance(exc, NON_RETRYABLE_EXCEPTIONS):
        return False

    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True

    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError, ConnectionError, TimeoutError)):
        return True

    if isinstance(exc, APIStatusError):
        code = exc.status_code
        if code in NON_RETRYABLE_STATUS_CODES:
            return False
        if code == 429 or code >= 500:
            return True

    return False


def extract_retry_after(exc: Exception) -> Optional[float]:
    """Extract Retry-After value in seconds if the exception or response exposes one."""
    # Direct attribute on custom or wrapped exceptions
    if hasattr(exc, "retry_after") and isinstance(getattr(exc, "retry_after"), (int, float)):
        return float(getattr(exc, "retry_after"))

    # Response header in APIStatusError / RateLimitError
    response = getattr(exc, "response", None)
    if response is not None and hasattr(response, "headers"):
        header_val = response.headers.get("retry-after") or response.headers.get("Retry-After")
        if header_val:
            try:
                return max(0.0, float(header_val))
            except ValueError:
                pass
    return None


def calculate_delay(
    attempt: int,
    base: float = 0.5,
    cap: float = 8.0,
    jitter_max: float = 0.25,
    retry_after: Optional[float] = None,
) -> float:
    """Calculate backoff delay: min(base * 2^attempt, cap) + random jitter.

    If Retry-After is present, honors that value with slight jitter to prevent lockstep.
    """
    if retry_after is not None:
        return retry_after + random.uniform(0.0, jitter_max)

    exponential = min(base * (2.0 ** attempt), cap)
    jitter = random.uniform(0.0, jitter_max)
    return exponential + jitter


@dataclass(frozen=True)
class RetryResult(Generic[T]):
    """Result of an operation executed through with_retry.

    Unpacks cleanly as `(result, attempts)` or accessed via attributes.
    """
    result: T
    attempts: int

    def __iter__(self):
        yield self.result
        yield self.attempts


async def with_retry(
    operation: Callable[[], Awaitable[T]],
    max_attempts: int = 4,
    base_delay: float = 0.5,
    cap_delay: float = 8.0,
    jitter_max: float = 0.25,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> RetryResult[T]:
    """Execute an async operation with exponential backoff and jitter.

    Args:
        operation: Async callable returning a value.
        max_attempts: Maximum number of attempts before giving up (default: 4).
        base_delay: Base delay in seconds.
        cap_delay: Maximum delay ceiling in seconds.
        jitter_max: Upper bound for random jitter added to delay.
        on_retry: Optional callback `(attempt_num, exception, delay)` invoked on each retry.

    Returns:
        RetryResult containing `result` and `attempts` (number of attempts used).

    Raises:
        The original exception immediately on non-retryable errors, or after
        max_attempts are exhausted.
    """
    attempt = 1
    while True:
        try:
            val = await operation()
            return RetryResult(result=val, attempts=attempt)
        except Exception as exc:
            # Check if error is non-retryable OR if we have exhausted all attempts
            if not is_retryable_error(exc):
                logger.warning(
                    f"Non-retryable error {type(exc).__name__} encountered on attempt {attempt}. "
                    "Giving up immediately."
                )
                raise exc

            if attempt >= max_attempts:
                logger.error(
                    f"Max attempts ({max_attempts}) exhausted. Giving up cleanly. "
                    f"Last error: {type(exc).__name__}: {exc}"
                )
                raise exc

            retry_after = extract_retry_after(exc)
            delay = calculate_delay(
                attempt=attempt - 1,  # 0-indexed for backoff power: 2^0, 2^1, ...
                base=base_delay,
                cap=cap_delay,
                jitter_max=jitter_max,
                retry_after=retry_after,
            )

            logger.info(
                f"Attempt {attempt}/{max_attempts} failed with {type(exc).__name__}. "
                f"Retrying in {delay:.2f}s..."
            )

            if on_retry is not None:
                on_retry(attempt, exc, delay)

            await asyncio.sleep(delay)
            attempt += 1

