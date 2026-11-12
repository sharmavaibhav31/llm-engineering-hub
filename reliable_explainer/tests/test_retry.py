"""Tests for reliable_explainer/retry.py."""

import pytest
import httpx
from groq import (
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from reliable_explainer.retry import (
    calculate_delay,
    extract_retry_after,
    is_retryable_error,
    with_retry,
)


@pytest.fixture
def dummy_request():
    return httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


@pytest.mark.asyncio
async def test_fails_twice_then_succeeds_with_growing_waits(dummy_request):
    """A fake function that fails twice then succeeds is retried exactly twice with growing waits."""
    call_count = 0
    recorded_delays = []

    def on_retry(attempt, exc, delay):
        recorded_delays.append((attempt, delay))

    async def flaky_op():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RateLimitError("Rate limited", response=httpx.Response(429, request=dummy_request), body=None)
        if call_count == 2:
            raise InternalServerError("Server error", response=httpx.Response(500, request=dummy_request), body=None)
        return "success_payload"

    result, attempts = await with_retry(
        flaky_op,
        max_attempts=4,
        base_delay=0.1,
        cap_delay=5.0,
        jitter_max=0.05,
        on_retry=on_retry,
    )

    assert result == "success_payload"
    assert attempts == 3
    assert call_count == 3
    assert len(recorded_delays) == 2

    # Verify retried exactly twice
    assert recorded_delays[0][0] == 1  # retry after attempt 1
    assert recorded_delays[1][0] == 2  # retry after attempt 2

    # Verify growing waits: base * 2^0 vs base * 2^1
    # attempt 0 base=0.1 (0.10 - 0.15), attempt 1 base=0.2 (0.20 - 0.25)
    assert recorded_delays[1][1] > recorded_delays[0][1]


@pytest.mark.asyncio
async def test_401_auth_error_is_not_retried(dummy_request):
    """A 401 AuthenticationError must not be retried, failing immediately on attempt 1."""
    call_count = 0

    async def auth_failing_op():
        nonlocal call_count
        call_count += 1
        raise AuthenticationError("Invalid API key", response=httpx.Response(401, request=dummy_request), body=None)

    with pytest.raises(AuthenticationError):
        await with_retry(auth_failing_op, max_attempts=4, base_delay=0.01)

    assert call_count == 1


@pytest.mark.asyncio
async def test_attempt_4_gives_up_cleanly(dummy_request):
    """When an operation fails repeatedly with retryable errors, attempt 4 gives up cleanly."""
    call_count = 0
    retries_recorded = []

    async def always_failing_op():
        nonlocal call_count
        call_count += 1
        raise RateLimitError("Rate limit exceeded", response=httpx.Response(429, request=dummy_request), body=None)

    with pytest.raises(RateLimitError):
        await with_retry(
            always_failing_op,
            max_attempts=4,
            base_delay=0.01,
            jitter_max=0.005,
            on_retry=lambda attempt, exc, delay: retries_recorded.append(attempt),
        )

    # 4 total attempts made before giving up cleanly
    assert call_count == 4
    assert retries_recorded == [1, 2, 3]


@pytest.mark.asyncio
async def test_honors_retry_after_header(dummy_request):
    """Respects Retry-After header if the exception exposes one."""
    call_count = 0
    observed_delays = []

    resp = httpx.Response(429, headers={"retry-after": "1.75"}, request=dummy_request)
    err = RateLimitError("Rate limited", response=resp, body=None)

    async def rate_limited_then_ok():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise err
        return "ok"

    result = await with_retry(
        rate_limited_then_ok,
        max_attempts=4,
        base_delay=0.1,
        jitter_max=0.1,
        on_retry=lambda attempt, exc, delay: observed_delays.append(delay),
    )

    assert result.result == "ok"
    assert result.attempts == 2
    assert len(observed_delays) == 1
    # Must be at least 1.75 seconds (1.75 + jitter up to 0.1)
    assert 1.75 <= observed_delays[0] <= 1.86


def test_is_retryable_classification(dummy_request):
    """Verify classification of retryable vs non-retryable errors."""
    # Non-retryable
    assert is_retryable_error(BadRequestError("bad req", response=httpx.Response(400, request=dummy_request), body=None)) is False
    assert is_retryable_error(AuthenticationError("auth", response=httpx.Response(401, request=dummy_request), body=None)) is False
    assert is_retryable_error(PermissionDeniedError("denied", response=httpx.Response(403, request=dummy_request), body=None)) is False
    assert is_retryable_error(NotFoundError("not found", response=httpx.Response(404, request=dummy_request), body=None)) is False

    # Retryable
    assert is_retryable_error(RateLimitError("rate limit", response=httpx.Response(429, request=dummy_request), body=None)) is True
    assert is_retryable_error(InternalServerError("server err", response=httpx.Response(500, request=dummy_request), body=None)) is True
    assert is_retryable_error(ConnectionError("socket dropped")) is True
    assert is_retryable_error(TimeoutError("timeout")) is True

