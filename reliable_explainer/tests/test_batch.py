"""Tests for reliable_explainer/batch.py."""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock
from groq import AuthenticationError
import httpx

from reliable_explainer.config import ExplainerConfig
from reliable_explainer.batch import run_batch, BatchItemResult


@pytest.fixture
def mock_config():
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    return ExplainerConfig(
        api_key="gsk_test_key",
        model="openai/gpt-oss-20b",
        client=client,
    )


@pytest.mark.asyncio
async def test_empty_list_returns_empty(mock_config):
    """Empty questions list returns an empty list without making API calls."""
    results = await run_batch([], concurrency=3, config=mock_config)
    assert results == []
    mock_config.client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_concurrency_limit_holds(mock_config):
    """A concurrency limit of 3 visibly holds; peak active tasks never exceeds 3."""
    current_active = 0
    peak_active = 0

    async def fake_create(*args, **kwargs):
        nonlocal current_active, peak_active
        current_active += 1
        peak_active = max(peak_active, current_active)
        await asyncio.sleep(0.04)
        current_active -= 1

        mock_choice = MagicMock()
        mock_choice.message.content = "Answer"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.total_tokens = 42
        return mock_resp

    mock_config.client.chat.completions.create = AsyncMock(side_effect=fake_create)

    questions = [f"Question {i}" for i in range(5)]
    results = await run_batch(questions, concurrency=3, config=mock_config, print_on_complete=False)

    assert len(results) == 5
    assert peak_active <= 3
    assert all(r.status == "success" for r in results)


@pytest.mark.asyncio
async def test_five_questions_finish_in_time_of_slowest_not_sum(mock_config):
    """Five questions finish in about the time of the slowest, not the sequential sum."""
    delay_per_item = 0.05

    async def fake_create(*args, **kwargs):
        await asyncio.sleep(delay_per_item)
        mock_choice = MagicMock()
        mock_choice.message.content = "Answer"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.total_tokens = 30
        return mock_resp

    mock_config.client.chat.completions.create = AsyncMock(side_effect=fake_create)

    questions = [f"Question {i}" for i in range(5)]

    start_time = time.perf_counter()
    # Concurrency 5 allows all 5 to run in parallel
    results = await run_batch(questions, concurrency=5, config=mock_config, print_on_complete=False)
    elapsed = time.perf_counter() - start_time

    assert len(results) == 5
    # Sequential sum would be >= 0.25s. Concurrent time will be ~0.05s - 0.12s
    assert elapsed < 0.20
    assert all(r.status == "success" for r in results)


@pytest.mark.asyncio
async def test_one_failure_does_not_cancel_rest(mock_config):
    """If one task fails (e.g. auth error or invalid input), other tasks complete successfully."""
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")

    async def fake_create(*args, **kwargs):
        messages = kwargs.get("messages", [])
        user_msg = messages[1]["content"] if len(messages) > 1 else ""

        if user_msg == "Fail Question":
            raise AuthenticationError("Auth failed", response=httpx.Response(401, request=req), body=None)

        mock_choice = MagicMock()
        mock_choice.message.content = f"Answer to {user_msg}"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.total_tokens = 50
        return mock_resp

    mock_config.client.chat.completions.create = AsyncMock(side_effect=fake_create)

    questions = ["Good Question 1", "Fail Question", "Good Question 2", "   "]
    results = await run_batch(questions, concurrency=2, config=mock_config, print_on_complete=False)

    assert len(results) == 4

    # Question 1: success
    assert results[0].question == "Good Question 1"
    assert results[0].status == "success"
    assert results[0].answer == "Answer to Good Question 1"
    assert results[0].tokens == 50

    # Question 2: failed due to auth error, but did not cancel Question 3
    assert results[1].question == "Fail Question"
    assert results[1].status == "failed"
    assert results[1].answer is None
    assert "AuthenticationError" in results[1].error

    # Question 3: success
    assert results[2].question == "Good Question 2"
    assert results[2].status == "success"
    assert results[2].answer == "Answer to Good Question 2"

    # Question 4: whitespace rejection
    assert results[3].question == "   "
    assert results[3].status == "failed"
    assert "whitespace" in results[3].error

