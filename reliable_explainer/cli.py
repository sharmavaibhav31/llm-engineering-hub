"""Unified Command Line Interface for Reliable Explainer.

Supports single-question streaming with retry telemetry, batch mode with
bounded concurrency, model overrides, and rich summary tables.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import List, Optional

from reliable_explainer.batch import run_batch
from reliable_explainer.config import DEFAULT_MODEL, ExplainerConfig, load_config
from reliable_explainer.explainer import StreamResult, explain_stream
from reliable_explainer.format import (
    Result,
    format_answer,
    format_question,
    format_retry_notice,
    format_summary_table,
)
from reliable_explainer.retry import with_retry


async def handle_single_question(
    question: str,
    config: ExplainerConfig,
    use_color: bool = True,
) -> Result:
    """Execute a single question with live streaming, retries, and formatting."""
    print(format_question(question))

    attempts_made = 1
    retries_info: List[str] = []

    def on_retry(attempt: int, exc: Exception, delay: float) -> None:
        nonlocal attempts_made
        attempts_made = attempt + 1
        notice = format_retry_notice(attempt, type(exc).__name__, delay)
        retries_info.append(notice)
        sys.stdout.write(f"\n{notice}\n")
        sys.stdout.flush()

    start_time = time.perf_counter()
    stream_res: Optional[StreamResult] = None
    error_msg: Optional[str] = None
    status = "success"

    try:
        async def _stream_op():
            return await explain_stream(question, config, print_stream=True)

        retry_res = await with_retry(_stream_op, max_attempts=4, on_retry=on_retry)
        stream_res = retry_res.result
        attempts_made = retry_res.attempts

    except Exception as exc:
        status = "failed"
        error_msg = f"{type(exc).__name__}: {exc}"
        sys.stdout.write(f"\n[FAILED]: {error_msg}\n")
        sys.stdout.flush()

    total_seconds = time.perf_counter() - start_time

    res = Result(
        question=question,
        status=status,
        attempts=attempts_made,
        seconds=round(total_seconds, 2),
        answer=stream_res.text if stream_res else None,
        tokens=stream_res.total_tokens if stream_res else None,
        retries_info=retries_info,
        error=error_msg,
    )

    print(format_summary_table([res], use_color=use_color))
    return res


async def handle_batch_questions(
    questions: List[str],
    concurrency: int,
    config: ExplainerConfig,
    use_color: bool = True,
) -> List[Result]:
    """Execute multiple questions concurrently with bounded concurrency."""
    print(f"\n--- Running Batch Mode ({len(questions)} questions, concurrency={concurrency}) ---")
    start_time = time.perf_counter()

    batch_items = await run_batch(
        questions=questions,
        concurrency=concurrency,
        config=config,
        print_on_complete=True,
    )

    total_seconds = time.perf_counter() - start_time
    # Approximate per-item average wall-clock duration for summary
    duration_per_item = total_seconds / len(questions) if questions else 0.0

    results: List[Result] = []
    for item in batch_items:
        results.append(
            Result(
                question=item.question,
                status=item.status,
                attempts=item.attempts,
                seconds=round(duration_per_item, 2),
                answer=item.answer,
                tokens=item.tokens,
                error=item.error,
            )
        )

    print(f"\n--- Batch Finished in {total_seconds:.2f}s ---")
    print(format_summary_table(results, use_color=use_color))
    return results


def build_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="explainer",
        description="Reliable Explainer: Production-grade streaming and resilient LLM client.",
    )
    parser.add_argument(
        "questions",
        nargs="*",
        help="One or more technical questions to explain. If omitted, prompts interactively.",
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help=f"Model identifier override (default: {DEFAULT_MODEL} or EXPLAINER_MODEL env var).",
    )
    parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=3,
        help="Concurrency limit for batch questions (default: 3).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Force batch mode even if a single question is provided.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output in summary tables.",
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    """Async main routine."""
    cfg = load_config(model_override=args.model)
    use_color = not args.no_color

    questions = args.questions
    if not questions:
        # Interactive mode
        try:
            sys.stdout.write("Enter your question: ")
            sys.stdout.flush()
            user_input = sys.stdin.readline()
            if not user_input or not user_input.strip():
                sys.stderr.write("Error: Question cannot be empty.\n")
                return 1
            questions = [user_input.strip()]
        except (KeyboardInterrupt, EOFError):
            sys.stdout.write("\nAborted.\n")
            return 0

    if len(questions) == 1 and not args.batch:
        res = await handle_single_question(questions[0], config=cfg, use_color=use_color)
        return 0 if res.status == "success" else 1
    else:
        results = await handle_batch_questions(
            questions=questions,
            concurrency=args.concurrency,
            config=cfg,
            use_color=use_color,
        )
        return 0 if any(r.status == "success" for r in results) else 1


def main() -> None:
    """CLI sync entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        exit_code = asyncio.run(async_main(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        sys.stderr.write("\nOperation interrupted by user.\n")
        sys.exit(130)


if __name__ == "__main__":
    main()
