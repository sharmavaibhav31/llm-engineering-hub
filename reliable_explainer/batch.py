"""Batch question explainer with bounded concurrency and fault isolation.

Uses asyncio.Semaphore to cap in-flight calls, asyncio.gather for parallel dispatch,
and with_retry for per-task resilience. One failing task never cancels the rest.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import List, Literal, Optional

from reliable_explainer.config import ExplainerConfig, load_config
from reliable_explainer.explainer import SYSTEM_PROMPT
from reliable_explainer.retry import with_retry


@dataclass(frozen=True)
class BatchItemResult:
    """Result of a single question execution in batch mode."""
    question: str
    status: Literal["success", "failed"]
    attempts: int
    answer: Optional[str]
    tokens: Optional[int]
    error: Optional[str] = None


# Alias for compatibility with section 5 telemetry
Result = BatchItemResult


async def run_batch(
    questions: List[str],
    concurrency: int = 3,
    config: Optional[ExplainerConfig] = None,
    print_on_complete: bool = True,
) -> List[BatchItemResult]:
    """Execute a batch of questions concurrently with bounded concurrency.

    Args:
        questions: List of question strings to explain.
        concurrency: Maximum number of concurrent requests allowed (default: 3).
        config: Optional ExplainerConfig instance (loads from env if None).
        print_on_complete: Whether to print each answer as it completes.

    Returns:
        List of BatchItemResult objects preserving input question order.
    """
    # 1. Handle empty list cleanly
    if not questions:
        return []

    cfg = config or load_config()
    semaphore = asyncio.Semaphore(concurrency)

    async def process_question(question: str) -> BatchItemResult:
        cleaned = question.strip() if question else ""
        if not cleaned:
            res = BatchItemResult(
                question=question,
                status="failed",
                attempts=0,
                answer=None,
                tokens=None,
                error="Question cannot be empty or whitespace-only.",
            )
            if print_on_complete:
                sys.stdout.write(f"\n[Q]: {question}\n[FAILED] (attempts: 0): {res.error}\n")
                sys.stdout.flush()
            return res

        attempts_made = 1

        def on_retry_hook(attempt: int, exc: Exception, delay: float) -> None:
            nonlocal attempts_made
            attempts_made = attempt + 1

        try:
            # Bound in-flight requests using the semaphore
            async with semaphore:
                async def _call():
                    return await cfg.client.chat.completions.create(
                        model=cfg.model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": cleaned},
                        ],
                        stream=False,  # Batch mode does not stream tokens
                    )

                retry_res = await with_retry(
                    _call,
                    max_attempts=4,
                    on_retry=on_retry_hook,
                )
                attempts_made = retry_res.attempts
                response = retry_res.result

                answer = response.choices[0].message.content or ""
                tokens = (
                    getattr(response.usage, "total_tokens", None)
                    if getattr(response, "usage", None)
                    else None
                )

                item_result = BatchItemResult(
                    question=question,
                    status="success",
                    attempts=attempts_made,
                    answer=answer,
                    tokens=tokens,
                )

        except Exception as exc:
            # Task-level error isolation: one failing question never cancels the batch
            item_result = BatchItemResult(
                question=question,
                status="failed",
                attempts=attempts_made,
                answer=None,
                tokens=None,
                error=f"{type(exc).__name__}: {exc}",
            )

        if print_on_complete:
            if item_result.status == "success":
                sys.stdout.write(
                    f"\n[Q]: {item_result.question}\n"
                    f"[A] (attempts: {item_result.attempts}, tokens: {item_result.tokens or 'n/a'}):\n"
                    f"{item_result.answer}\n"
                )
            else:
                sys.stdout.write(
                    f"\n[Q]: {item_result.question}\n"
                    f"[FAILED] (attempts: {item_result.attempts}): {item_result.error}\n"
                )
            sys.stdout.flush()

        return item_result

    # Dispatch all tasks concurrently via asyncio.gather
    tasks = [process_question(q) for q in questions]
    results = await asyncio.gather(*tasks)

    return list(results)

