"""Single-question streaming explainer.

Streams LLM completions token-by-token, filtering empty deltas,
and measuring Time To First Token (TTFT) and total duration.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional

from reliable_explainer.config import ExplainerConfig

SYSTEM_PROMPT = (
    "You are an expert technical explainer. Provide concise, clear, and accurate "
    "technical explanations suitable for a backend or systems engineer. "
    "Avoid unnecessary filler or repetitive pleasantries."
)


@dataclass(frozen=True)
class StreamResult:
    """Telemetry and output result from a single streaming explanation."""
    text: str
    ttft_seconds: float
    total_seconds: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


async def explain_stream(
    question: str,
    config: ExplainerConfig,
    print_stream: bool = True,
) -> StreamResult:
    """Stream an explanation for a single question.

    Args:
        question: The user prompt to explain.
        config: ExplainerConfig instance containing the client and model.
        print_stream: Whether to flush chunks directly to sys.stdout as they arrive.

    Returns:
        StreamResult containing the collected text, ttft_seconds, total_seconds,
        and token usage if provided by the model.

    Raises:
        ValueError: If question is empty or whitespace-only (rejected prior to API call).
    """
    cleaned_question = question.strip() if question else ""
    if not cleaned_question:
        raise ValueError("Question cannot be empty or whitespace-only.")

    start_time = time.perf_counter()
    first_token_time: Optional[float] = None
    chunks: list[str] = []
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Call the Groq SDK async completions with streaming enabled
    stream = await config.client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": cleaned_question},
        ],
        stream=True,
    )

    async for chunk in stream:
        # Extract delta text if choice is present
        delta = chunk.choices[0].delta.content if chunk.choices else None

        # Check for token usage metadata on the chunk
        usage = getattr(chunk, "usage", None) or getattr(getattr(chunk, "x_groq", None), "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)

        # Skip empty deltas (None, empty strings, reasoning metadata)
        if not delta:
            continue

        # Record TTFT on the exact arrival of the first non-empty text chunk
        if first_token_time is None:
            first_token_time = time.perf_counter()

        chunks.append(delta)

        if print_stream:
            sys.stdout.write(delta)
            sys.stdout.flush()

    if print_stream and chunks:
        sys.stdout.write("\n")
        sys.stdout.flush()

    end_time = time.perf_counter()
    total_seconds = end_time - start_time
    ttft_seconds = (first_token_time - start_time) if first_token_time is not None else total_seconds

    return StreamResult(
        text="".join(chunks),
        ttft_seconds=round(ttft_seconds, 4),
        total_seconds=round(total_seconds, 4),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )

