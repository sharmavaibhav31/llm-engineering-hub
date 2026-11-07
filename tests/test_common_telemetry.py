"""Tests for common/telemetry.py logic."""

import time
from common.telemetry import TelemetryTracker, RequestTelemetry


def test_telemetry_tracker_lifecycle():
    tracker = TelemetryTracker(prompt="Explain backoff", model="llama-3.3-70b-versatile")
    time.sleep(0.01)
    tracker.record_first_token()
    time.sleep(0.01)
    tracker.record_usage(prompt_tokens=15, completion_tokens=50, total_tokens=65)
    rec = tracker.finalize()

    assert rec.prompt == "Explain backoff"
    assert rec.model == "llama-3.3-70b-versatile"
    assert rec.status == "success"
    assert rec.attempts == 1
    assert rec.ttft_ms is not None
    assert rec.ttft_ms > 0
    assert rec.total_duration_ms >= rec.ttft_ms
    assert rec.prompt_tokens == 15
    assert rec.completion_tokens == 50
    assert rec.total_tokens == 65
    assert rec.tokens_per_second is not None
    assert rec.tokens_per_second > 0


def test_telemetry_retry_recording():
    tracker = TelemetryTracker(prompt="Failing call", model="llama-3.3-70b-versatile")
    tracker.record_retry(attempt=1, exc=ValueError("network drop"), delay_seconds=0.5)
    tracker.mark_failure(ValueError("network drop"))
    rec = tracker.finalize()

    assert rec.status == "failed"
    assert rec.attempts == 2
    assert len(rec.retry_history) == 1
    assert rec.retry_history[0].delay_seconds == 0.5
    assert rec.retry_history[0].error_type == "ValueError"

