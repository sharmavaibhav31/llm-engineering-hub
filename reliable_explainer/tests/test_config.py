"""Tests for reliable_explainer/config.py."""

import pytest
from unittest.mock import patch
from reliable_explainer.config import load_config, DEFAULT_MODEL


def test_missing_groq_api_key_exits_with_error(monkeypatch, capsys):
    """Missing or empty GROQ_API_KEY must print one clear message and exit non-zero."""
    # Ensure GROQ_API_KEY is unset and load_dotenv does not reload it from .env file
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with patch("reliable_explainer.config.load_dotenv"):
        with pytest.raises(SystemExit) as exc_info:
            load_config()

        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "Error: GROQ_API_KEY environment variable is missing or empty." in captured.err


def test_model_resolution_order(monkeypatch):
    """Test model hierarchy: CLI flag > EXPLAINER_MODEL env var > DEFAULT_MODEL."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_dummy_test_key_12345")

    with patch("reliable_explainer.config.load_dotenv"):
        # 1. Default fallback when EXPLAINER_MODEL is unset
        monkeypatch.delenv("EXPLAINER_MODEL", raising=False)
        cfg_default = load_config()
        assert cfg_default.model == DEFAULT_MODEL
        assert cfg_default.client.max_retries == 0

        # 2. Env var takes effect when set
        monkeypatch.setenv("EXPLAINER_MODEL", "llama-3.1-8b-instant")
        cfg_env = load_config()
        assert cfg_env.model == "llama-3.1-8b-instant"

        # 3. CLI flag override takes precedence over env var
        cfg_override = load_config(model_override="mixtral-8x7b-32768")
        assert cfg_override.model == "mixtral-8x7b-32768"

