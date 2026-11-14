"""Tests for lcel_pipeline CLI error boundaries and exception mapping."""

import inspect
import pytest
from unittest.mock import patch
import httpx
from groq import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from lcel_pipeline.errors import (
    AUTH_ERROR_MSG,
    NETWORK_ERROR_MSG,
    RATE_LIMIT_ERROR_MSG,
    handle_cli_error,
)
from lcel_pipeline.cli import main
import lcel_pipeline.explain as explain_mod


@pytest.fixture
def dummy_request():
    return httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def test_auth_error_branch(dummy_request, capsys):
    """AuthenticationError maps to the distinct auth error message."""
    resp = httpx.Response(401, request=dummy_request)
    err = AuthenticationError("Invalid API key", response=resp, body=None)

    code = handle_cli_error(err)
    assert code == 1

    captured = capsys.readouterr()
    assert AUTH_ERROR_MSG in captured.err
    assert "Traceback" not in captured.err


def test_rate_limit_error_branch(dummy_request, capsys):
    """RateLimitError maps to the distinct rate limit error message."""
    resp = httpx.Response(429, request=dummy_request)
    err = RateLimitError("Rate limit exceeded", response=resp, body=None)

    code = handle_cli_error(err)
    assert code == 1

    captured = capsys.readouterr()
    assert RATE_LIMIT_ERROR_MSG in captured.err
    assert "Traceback" not in captured.err


def test_network_connection_error_branch(dummy_request, capsys):
    """APIConnectionError and APITimeoutError map to the distinct network error message."""
    err_conn = APIConnectionError(request=dummy_request)
    code1 = handle_cli_error(err_conn)
    assert code1 == 1
    captured1 = capsys.readouterr()
    assert NETWORK_ERROR_MSG in captured1.err

    err_timeout = APITimeoutError(request=dummy_request)
    code2 = handle_cli_error(err_timeout)
    assert code2 == 1
    captured2 = capsys.readouterr()
    assert NETWORK_ERROR_MSG in captured2.err


def test_cli_boundary_catches_auth_failure(monkeypatch, dummy_request, capsys):
    """When explain execution encounters 401, CLI prints auth message without traceback."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_invalid_test_key")
    resp = httpx.Response(401, request=dummy_request)
    auth_err = AuthenticationError("Unauthorized", response=resp, body=None)

    with patch("lcel_pipeline.config.load_dotenv"):
        with patch("lcel_pipeline.explain.run_explain", side_effect=auth_err):
            exit_code = main(["explain", "Backoff"])
            assert exit_code == 1

            captured = capsys.readouterr()
            assert AUTH_ERROR_MSG in captured.err
            assert "Traceback" not in captured.err


def test_cli_boundary_catches_rate_limit(monkeypatch, dummy_request, capsys):
    """When explain execution encounters 429, CLI prints rate limit message without traceback."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    resp = httpx.Response(429, request=dummy_request)
    rate_err = RateLimitError("Too Many Requests", response=resp, body=None)

    with patch("lcel_pipeline.config.load_dotenv"):
        with patch("lcel_pipeline.explain.run_explain", side_effect=rate_err):
            exit_code = main(["explain", "Backoff"])
            assert exit_code == 1

            captured = capsys.readouterr()
            assert RATE_LIMIT_ERROR_MSG in captured.err
            assert "Traceback" not in captured.err


def test_cli_boundary_catches_network_failure(monkeypatch, dummy_request, capsys):
    """When explain execution encounters connection drop, CLI prints network message without traceback."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    conn_err = APIConnectionError(request=dummy_request)

    with patch("lcel_pipeline.config.load_dotenv"):
        with patch("lcel_pipeline.explain.run_explain", side_effect=conn_err):
            exit_code = main(["explain", "Backoff"])
            assert exit_code == 1

            captured = capsys.readouterr()
            assert NETWORK_ERROR_MSG in captured.err
            assert "Traceback" not in captured.err


def test_exception_handling_lives_at_cli_edge_not_inside_chain():
    """Verify that explain.py does NOT swallow or handle provider API exceptions."""
    explain_source = inspect.getsource(explain_mod.run_explain)
    assert "handle_cli_error" not in explain_source
    assert "AuthenticationError" not in explain_source
    assert "RateLimitError" not in explain_source
    assert "APIConnectionError" not in explain_source

