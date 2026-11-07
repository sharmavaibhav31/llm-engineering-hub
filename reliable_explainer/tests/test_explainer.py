"""Tests for reliable_explainer/explainer.py."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from reliable_explainer.config import ExplainerConfig
from reliable_explainer.explainer import explain_stream, StreamResult


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
async def test_empty_or_whitespace_question_rejected(mock_config):
    """Empty or whitespace-only input must raise ValueError before any API call."""
    for empty_input in ["", "   ", "\n\t  ", None]:
        with pytest.raises(ValueError, match="Question cannot be empty or whitespace-only"):
            await explain_stream(empty_input, mock_config)

    # Verify no API call was dispatched
    mock_config.client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_explain_stream_skips_empty_deltas_and_tracks_time(mock_config):
    """Stream chunks with None and empty string must be skipped; TTFT and text must be tracked."""
    class DummyDelta:
        def __init__(self, content):
            self.content = content

    class DummyChoice:
        def __init__(self, content):
            self.delta = DummyDelta(content)

    class DummyUsage:
        prompt_tokens = 10
        completion_tokens = 25
        total_tokens = 35

    class DummyChunk:
        def __init__(self, content, usage=None):
            self.choices = [DummyChoice(content)] if content is not None else []
            self.usage = usage

    async def fake_stream_generator():
        # Yield empty / metadata chunks first
        yield DummyChunk("")
        yield DummyChunk(None)
        # Yield actual content tokens
        yield DummyChunk("Hello")
        yield DummyChunk(" world")
        yield DummyChunk("!")
        # Yield final usage chunk
        yield DummyChunk("", usage=DummyUsage())

    mock_config.client.chat.completions.create = AsyncMock(return_value=fake_stream_generator())

    result = await explain_stream("What is async?", mock_config, print_stream=False)

    assert isinstance(result, StreamResult)
    assert result.text == "Hello world!"
    assert result.ttft_seconds >= 0.0
    assert result.total_seconds >= result.ttft_seconds
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 25
    assert result.total_tokens == 35

    # Verify create was called with system and user messages
    mock_config.client.chat.completions.create.assert_called_once()
    call_kwargs = mock_config.client.chat.completions.create.call_args.kwargs
    assert call_kwargs["stream"] is True
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["role"] == "user"
    assert call_kwargs["messages"][1]["content"] == "What is async?"

