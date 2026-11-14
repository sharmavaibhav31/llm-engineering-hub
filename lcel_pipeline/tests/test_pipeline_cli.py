"""Tests for lcel_pipeline CLI skeleton and configuration."""

import pytest
from unittest.mock import patch
from lcel_pipeline.cli import build_parser, main
from lcel_pipeline.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    load_settings,
    validate_chunk_settings,
)


def test_explain_subcommand_parses():
    """Verify explain subcommand parses all arguments and defaults."""
    parser = build_parser()
    args = parser.parse_args([
        "explain", "Prompt Engineering",
        "-a", "beginners",
        "-f", "bullet points",
        "--show-prompt",
    ])
    assert args.subcommand == "explain"
    assert args.topic == "Prompt Engineering"
    assert args.audience == "beginners"
    assert args.format == "bullet points"
    assert args.show_prompt is True
    assert args.chunk_size == DEFAULT_CHUNK_SIZE
    assert args.chunk_overlap == DEFAULT_CHUNK_OVERLAP


def test_document_subcommand_parses():
    """Verify document subcommand parses file path and chunk flags."""
    parser = build_parser()
    args = parser.parse_args([
        "document", "notes.txt",
        "--chunk-size", "1000",
        "--chunk-overlap", "150",
        "-n", "5",
    ])
    assert args.subcommand == "document"
    assert args.file_path == "notes.txt"
    assert args.chunk_size == 1000
    assert args.chunk_overlap == 150
    assert args.num_chunks == 5


def test_overlap_larger_than_size_gives_friendly_error(capsys):
    """Overlap larger than size must exit with code 1 and friendly message, not a stack trace."""
    with pytest.raises(SystemExit) as exc_info:
        main(["document", "notes.txt", "--chunk-size", "200", "--chunk-overlap", "300"])

    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "Error: chunk_overlap (300) cannot be larger than chunk_size (200)" in captured.err
    # Verify no unhandled traceback was printed
    assert "Traceback" not in captured.err


def test_missing_groq_api_key_fails_with_clear_message(monkeypatch, capsys):
    """Missing or empty GROQ_API_KEY must print one clear message and exit non-zero."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with patch("lcel_pipeline.config.load_dotenv"):
        with pytest.raises(SystemExit) as exc_info:
            main(["explain", "Backoff"])

        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "Error: GROQ_API_KEY environment variable is missing or empty" in captured.err
        assert "Traceback" not in captured.err


def test_valid_execution_skeleton(monkeypatch, capsys):
    """Skeleton executes successfully when settings and arguments are valid."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_dummy_12345")
    with patch("lcel_pipeline.config.load_dotenv"):
        code_explain = main(["explain", "Concurrency", "--show-prompt"])
        assert code_explain == 0
        captured = capsys.readouterr()
        assert "=== FORMATTED PROMPT (No API Call) ===" in captured.out
        assert "Topic:           Concurrency" in captured.out

        code_doc = main(["document", "README.md", "--chunk-size", "500", "--chunk-overlap", "50"])
        assert code_doc == 0
        captured_doc = capsys.readouterr()
        assert "DOCUMENT INGESTION & CHUNKING REPORT" in captured_doc.out
        assert "Chunk Size / Lap:  500 / 50" in captured_doc.out
