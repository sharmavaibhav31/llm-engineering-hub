"""Tests for common/retry.py logic."""

import pytest
import httpx
from unittest.mock import MagicMock
from groq import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
    InternalServerError,
)
from common.retry import (
    calculate_backoff,
    extract_retry_after,
    is_retryable_error,
    execute_with_retry,
)


def test_is_retryable_error():
    req = httpx.Request("POST", "https://api.groq.com")

    # Retryable cases
    rate_limit_err = RateLimitError("Rate limit exceeded", response=httpx.Response(429, request=req), body=None)
    assert is_retryable_error(rate_limit_err) is True

    server_err = InternalServerError("Internal Server Error", response=httpx.Response(500, request=req), body=None)
    assert is_retryable_error(server_err) is True

    gateway_timeout_err = InternalServerError("Gateway Timeout", response=httpx.Response(504, request=req), body=None)
    assert is_retryable_error(gateway_timeout_err) is True

    timeout_err = APITimeoutError(request=req)
    assert is_retryable_error(timeout_err) is True

    conn_err = APIConnectionError(request=req)
    assert is_retryable_error(conn_err) is True

    # Non-retryable cases
    bad_req_err = BadRequestError("Invalid prompt", response=httpx.Response(400, request=req), body=None)
    assert is_retryable_error(bad_req_err) is False

    auth_err = AuthenticationError("Invalid API Key", response=httpx.Response(401, request=req), body=None)
    assert is_retryable_error(auth_err) is False

    not_found_err = NotFoundError("Model not found", response=httpx.Response(404, request=req), body=None)
    assert is_retryable_error(not_found_err) is False


def test_extract_retry_after():
    req = httpx.Request("POST", "https://api.groq.com")
    resp_with_header = httpx.Response(429, headers={"retry-after": "4.2"}, request=req)
    err = RateLimitError("Rate limit", response=resp_with_header, body=None)
    assert extract_retry_after(err) == 4.2

    resp_without_header = httpx.Response(500, request=req)
    err2 = InternalServerError("Server error", response=resp_without_header, body=None)
    assert extract_retry_after(err2) is None


def test_calculate_backoff_full_jitter():
    # Attempt 0: cap is 0.5 * 2^0 = 0.5
    for _ in range(20):
        val = calculate_backoff(attempt=0, base_backoff=0.5, max_backoff=8.0)
        assert 0.0 <= val <= 0.5

    # Attempt 3: cap is 0.5 * 2^3 = 4.0
    for _ in range(20):
        val = calculate_backoff(attempt=3, base_backoff=0.5, max_backoff=8.0)
        assert 0.0 <= val <= 4.0

    # Respect max_backoff
    val = calculate_backoff(attempt=10, base_backoff=0.5, max_backoff=5.0)
    assert 0.0 <= val <= 5.0


def test_calculate_backoff_with_retry_after():
    delay = calculate_backoff(attempt=0, base_backoff=0.5, max_backoff=8.0, retry_after=3.0)
    # retry_after + jitter between 0.05 and 0.2
    assert 3.05 <= delay <= 3.25


@pytest.mark.asyncio
async def test_execute_with_retry_success():
    counter = 0

    async def flaky_op():
        nonlocal counter
        counter += 1
        return "success"

    result = await execute_with_retry(flaky_op, max_retries=3, base_backoff=0.01)
    assert result == "success"
    assert counter == 1


@pytest.mark.asyncio
async def test_execute_with_retry_recover_after_rate_limit():
    counter = 0
    req = httpx.Request("POST", "https://api.groq.com")
    retries_recorded = []

    def on_retry(attempt, exc, delay):
        retries_recorded.append((attempt, type(exc).__name__, delay))

    async def flaky_op():
        nonlocal counter
        counter += 1
        if counter < 3:
            raise RateLimitError("Rate limited", response=httpx.Response(429, request=req), body=None)
        return "recovered"

    result = await execute_with_retry(
        flaky_op,
        max_retries=3,
        base_backoff=0.01,
        on_retry=on_retry,
    )
    assert result == "recovered"
    assert counter == 3
    assert len(retries_recorded) == 2


@pytest.mark.asyncio
async def test_execute_with_retry_fails_on_non_retryable():
    counter = 0
    req = httpx.Request("POST", "https://api.groq.com")

    async def bad_request_op():
        nonlocal counter
        counter += 1
        raise BadRequestError("Bad parameters", response=httpx.Response(400, request=req), body=None)

    with pytest.raises(BadRequestError):
        await execute_with_retry(bad_request_op, max_retries=3, base_backoff=0.01)

    # Must NOT retry 400 Bad Request
    assert counter == 1

