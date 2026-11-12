"""Tests for reliable_explainer/format.py."""

import sys
import pytest
from reliable_explainer.format import (
    Result,
    format_answer,
    format_question,
    format_retry_notice,
    format_summary_table,
)


def test_format_has_no_api_dependencies():
    """format.py must have zero API or network client dependencies."""
    import reliable_explainer.format as fmt_mod
    imported_modules = dir(fmt_mod)

    assert "groq" not in imported_modules
    assert "httpx" not in imported_modules
    assert "AsyncGroq" not in imported_modules


def test_format_question_and_retry_and_answer():
    """Verify visual separation helpers."""
    q_out = format_question("What is backoff?")
    assert "[QUESTION]: What is backoff?" in q_out
    assert "=" in q_out

    retry_out = format_retry_notice(attempt=1, error_desc="RateLimitError", delay_seconds=1.25)
    assert "[RETRY NOTICE]" in retry_out
    assert "Attempt 1 failed (RateLimitError)" in retry_out
    assert "1.25s" in retry_out

    ans_out = format_answer("Backoff increases wait time exponentially.", status="success")
    assert "[ANSWER]:" in ans_out
    assert "Backoff increases wait time exponentially." in ans_out

    fail_out = format_answer(None, status="failed", error="401 Auth Failed")
    assert "[FAILED]:" in fail_out
    assert "401 Auth Failed" in fail_out


def test_summary_table_columns_and_formatting():
    """Summary table must list question, status, attempts, tokens, and seconds."""
    results = [
        Result(
            question="What is TTFT?",
            status="success",
            attempts=1,
            seconds=0.74,
            answer="Time To First Token.",
            tokens=128,
        ),
        Result(
            question="What is jitter?",
            status="success",
            attempts=3,
            seconds=2.45,
            answer="Randomized delay.",
            tokens=95,
        ),
    ]

    table_text = format_summary_table(results, use_color=False)

    assert "What is TTFT?" in table_text
    assert "SUCCESS" in table_text
    assert "128" in table_text
    assert "0.74s" in table_text
    assert "What is jitter?" in table_text
    assert "95" in table_text
    assert "2.45s" in table_text


def test_missing_usage_data_displays_na_without_crashing():
    """When token usage is None, summary table displays 'n/a' cleanly without crashing."""
    results = [
        Result(
            question="Why did stream fail?",
            status="failed",
            attempts=4,
            seconds=5.12,
            answer=None,
            tokens=None,  # Missing usage data
            error="Connection dropped",
        ),
        Result(
            question="Another query with no usage",
            status="success",
            attempts=1,
            seconds=0.55,
            answer="Some answer",
            tokens=None,  # Missing usage data
        ),
    ]

    # Must not raise exception
    table_text = format_summary_table(results, use_color=False)

    assert "n/a" in table_text
    assert "FAILED" in table_text
    assert "5.12s" in table_text
    assert "0.55s" in table_text


def test_empty_results_summary():
    """Formatting an empty results list returns clean placeholder without error."""
    out = format_summary_table([])
    assert "No results recorded" in out

