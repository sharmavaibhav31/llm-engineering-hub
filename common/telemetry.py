"""Structured telemetry and observability for LLM requests.

Captures TTFT (Time to First Token), total duration, retry attempts,
status, and optional token usage across streaming and non-streaming calls.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table


class RetryEvent(BaseModel):
    """Details of a single retry event."""
    attempt: int
    delay_seconds: float
    error_type: str
    error_message: str


class RequestTelemetry(BaseModel):
    """Structured telemetry record per LLM request."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    prompt: str
    model: str
    status: Literal["success", "failed", "partial_failure"] = "success"
    attempts: int = 1
    ttft_ms: Optional[float] = Field(
        default=None,
        description="Time To First Token in milliseconds (user-perceived latency)",
    )
    total_duration_ms: float = Field(
        default=0.0,
        description="Total round-trip latency in milliseconds",
    )
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None
    retry_history: List[RetryEvent] = Field(default_factory=list)
    error: Optional[str] = None

    def calculate_tokens_per_second(self) -> Optional[float]:
        """Compute generation speed in tokens per second."""
        if self.completion_tokens and self.total_duration_ms > 0:
            # If TTFT is known, generation time is total_time - ttft
            gen_time_s = (
                (self.total_duration_ms - (self.ttft_ms or 0)) / 1000.0
                if self.ttft_ms
                else self.total_duration_ms / 1000.0
            )
            if gen_time_s > 0:
                self.tokens_per_second = round(self.completion_tokens / gen_time_s, 2)
                return self.tokens_per_second
        return None


class TelemetryTracker:
    """Tracks latency, TTFT, retries, and token usage for an active LLM operation."""

    def __init__(self, prompt: str, model: str, request_id: Optional[str] = None):
        self.request_id = request_id or str(uuid.uuid4())[:8]
        self.prompt = prompt
        self.model = model
        self.start_time: float = time.perf_counter()
        self.first_token_time: Optional[float] = None
        self.attempts: int = 1
        self.retry_history: List[RetryEvent] = []
        self.prompt_tokens: Optional[int] = None
        self.completion_tokens: Optional[int] = None
        self.total_tokens: Optional[int] = None
        self.error: Optional[str] = None
        self.status: Literal["success", "failed", "partial_failure"] = "success"

    def record_first_token(self) -> None:
        """Call when the very first non-empty text token arrives from stream."""
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def record_retry(self, attempt: int, exc: Exception, delay_seconds: float) -> None:
        """Record a retry attempt and its backoff delay."""
        self.attempts = max(self.attempts, attempt + 1)
        self.retry_history.append(
            RetryEvent(
                attempt=attempt,
                delay_seconds=round(delay_seconds, 3),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        )

    def record_usage(
        self,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> None:
        """Extract and store token usage (from Groq final stream chunk or response)."""
        if prompt_tokens is not None:
            self.prompt_tokens = prompt_tokens
        if completion_tokens is not None:
            self.completion_tokens = completion_tokens
        if total_tokens is not None:
            self.total_tokens = total_tokens
        elif prompt_tokens is not None and completion_tokens is not None:
            self.total_tokens = prompt_tokens + completion_tokens

    def mark_failure(self, exc: Exception, is_partial: bool = False) -> None:
        """Record fatal failure."""
        self.status = "partial_failure" if is_partial else "failed"
        self.error = f"{type(exc).__name__}: {exc}"

    def finalize(self) -> RequestTelemetry:
        """Finalize timing calculations and return an immutable telemetry record."""
        end_time = time.perf_counter()
        total_duration_ms = (end_time - self.start_time) * 1000.0

        ttft_ms = None
        if self.first_token_time is not None:
            ttft_ms = (self.first_token_time - self.start_time) * 1000.0

        telemetry = RequestTelemetry(
            request_id=self.request_id,
            prompt=self.prompt,
            model=self.model,
            status=self.status,
            attempts=self.attempts,
            ttft_ms=round(ttft_ms, 2) if ttft_ms is not None else None,
            total_duration_ms=round(total_duration_ms, 2),
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            retry_history=self.retry_history,
            error=self.error,
        )
        telemetry.calculate_tokens_per_second()
        return telemetry


def display_telemetry(telemetry: RequestTelemetry, console: Optional[Console] = None) -> None:
    """Render a rich, formatted telemetry card in the terminal."""
    con = console or Console()
    table = Table(title=f"Telemetry Report [dim]({telemetry.request_id})[/dim]", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    status_color = "green" if telemetry.status == "success" else "red"
    table.add_row("Status", f"[{status_color}]{telemetry.status.upper()}[/{status_color}]")
    table.add_row("Model", telemetry.model)
    table.add_row("Attempts", str(telemetry.attempts))
    table.add_row("Total Latency", f"{telemetry.total_duration_ms:.1f} ms ({telemetry.total_duration_ms / 1000:.2f} s)")

    ttft_str = f"{telemetry.ttft_ms:.1f} ms" if telemetry.ttft_ms is not None else "N/A"
    table.add_row("Time to First Token (TTFT)", f"[bold yellow]{ttft_str}[/bold yellow]")

    prompt_tok = str(telemetry.prompt_tokens) if telemetry.prompt_tokens is not None else "N/A"
    comp_tok = str(telemetry.completion_tokens) if telemetry.completion_tokens is not None else "N/A"
    total_tok = str(telemetry.total_tokens) if telemetry.total_tokens is not None else "N/A"

    table.add_row("Prompt / Completion / Total Tokens", f"{prompt_tok} / {comp_tok} / {total_tok}")

    if telemetry.tokens_per_second:
        table.add_row("Speed", f"{telemetry.tokens_per_second} tokens/sec")

    if telemetry.retry_history:
        retries_summary = ", ".join(
            f"Attempt {r.attempt}: {r.error_type} (waited {r.delay_seconds}s)"
            for r in telemetry.retry_history
        )
        table.add_row("Retry Events", f"[yellow]{retries_summary}[/yellow]")

    if telemetry.error:
        table.add_row("Error", f"[red]{telemetry.error}[/red]")

    con.print(table)

