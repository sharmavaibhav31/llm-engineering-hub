"""Formatting and telemetry reporting module.

Contains pure presentation and formatting logic with zero API or network dependencies.
Visually separates questions, retry notices, answers, and renders summary tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class Result:
    """Telemetry and outcome record for an executed question."""
    question: str
    status: str
    attempts: int
    seconds: float
    answer: Optional[str] = None
    tokens: Optional[int] = None
    retries_info: List[str] = field(default_factory=list)
    error: Optional[str] = None


def format_question(question: str) -> str:
    """Format a question header with visual separation."""
    separator = "=" * 60
    return f"\n{separator}\n[QUESTION]: {question}\n{separator}"


def format_retry_notice(attempt: int, error_desc: str, delay_seconds: float) -> str:
    """Format a retry notice banner."""
    return f"[RETRY NOTICE] Attempt {attempt} failed ({error_desc}). Retrying in {delay_seconds:.2f}s..."


def format_answer(answer: Optional[str], status: str = "success", error: Optional[str] = None) -> str:
    """Format the final answer or error block with visual separation."""
    separator = "-" * 60
    if status == "success" and answer is not None:
        return f"\n[ANSWER]:\n{answer.strip()}\n{separator}"
    return f"\n[FAILED]:\n{error or 'Unknown error'}\n{separator}"


def format_summary_table(results: List[Result], use_color: bool = True) -> str:
    """Render a formatted summary table for a collection of results.

    Columns: Question | Status | Attempts | Tokens | Seconds
    Handles missing tokens (None) by displaying 'n/a' without crashing.
    """
    if not results:
        return "\n--- SUMMARY TABLE: No results recorded ---\n"

    console = Console(record=True, color_system="auto" if use_color else None, width=100)
    table = Table(
        title="Execution Summary Table",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )

    table.add_column("Question", style="bold", overflow="fold", max_width=40)
    table.add_column("Status", justify="center")
    table.add_column("Attempts", justify="center")
    table.add_column("Tokens", justify="right")
    table.add_column("Seconds", justify="right")

    for item in results:
        # Question display
        q_display = item.question if len(item.question) <= 37 else item.question[:34] + "..."

        # Status with optional color
        if item.status == "success":
            status_display = "[green]SUCCESS[/green]" if use_color else "SUCCESS"
        else:
            status_display = "[red]FAILED[/red]" if use_color else "FAILED"

        # Attempts
        attempts_display = str(item.attempts)

        # Tokens: handle None gracefully as 'n/a'
        tokens_display = str(item.tokens) if item.tokens is not None else "n/a"

        # Seconds
        seconds_display = f"{item.seconds:.2f}s"

        table.add_row(
            q_display,
            status_display,
            attempts_display,
            tokens_display,
            seconds_display,
        )

    console.print(table)
    return console.export_text()

