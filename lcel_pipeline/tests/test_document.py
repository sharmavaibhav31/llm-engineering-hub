"""Tests for lcel_pipeline/document.py."""

import os
import tempfile
import pytest

from lcel_pipeline.config import PipelineSettings
from lcel_pipeline.document import (
    SplitSummary,
    load_and_split_document,
    format_document_output,
    run_document_mode,
)
from lcel_pipeline.cli import main


@pytest.fixture
def sample_text_file():
    content = (
        "Heading: Distributed Systems\n\n"
        "Paragraph 1: In distributed systems, consensus is critical. Nodes coordinate using protocols like Raft or Paxos.\n\n"
        "Paragraph 2: Network partitions and latency variations require robust retry and backoff strategies to prevent cascading failures.\n\n"
        "Paragraph 3: Bounded concurrency caps in-flight operations so the downstream service is never overwhelmed."
    )
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name

    yield path

    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def empty_text_file():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("   \n\t  \n  ")
        path = f.name

    yield path

    if os.path.exists(path):
        os.remove(path)


def test_load_and_split_document_success(sample_text_file):
    """Document is split into chunks with add_start_index metadata."""
    summary = load_and_split_document(
        file_path=sample_text_file,
        chunk_size=120,
        chunk_overlap=30,
    )

    assert isinstance(summary, SplitSummary)
    assert summary.total_characters > 0
    assert summary.total_chunks > 1
    assert len(summary.chunks) == summary.total_chunks

    # Verify each chunk has start_index in metadata
    for chunk in summary.chunks:
        assert "start_index" in chunk.metadata
        assert isinstance(chunk.metadata["start_index"], int)
        assert len(chunk.page_content) <= 120

    # Verify formatting output contains count, lengths, and start_index
    formatted = format_document_output(summary, num_chunks=2)
    assert "DOCUMENT INGESTION & CHUNKING REPORT" in formatted
    assert f"Total Chunks:      {summary.total_chunks}" in formatted
    assert "start_index:" in formatted
    assert "[Chunk 1/" in formatted
    assert "[Chunk 2/" in formatted


def test_missing_file_raises_friendly_error():
    """Missing file raises FileNotFoundError with clean message."""
    missing_path = "/tmp/non_existent_file_12345.txt"
    with pytest.raises(FileNotFoundError, match="File not found"):
        load_and_split_document(missing_path, chunk_size=800, chunk_overlap=120)


def test_empty_document_raises_friendly_error(empty_text_file):
    """Empty or whitespace-only document raises ValueError."""
    with pytest.raises(ValueError, match="is empty or whitespace-only"):
        load_and_split_document(empty_text_file, chunk_size=800, chunk_overlap=120)


def test_cli_document_mode_success(sample_text_file, capsys):
    """CLI document command prints chunk count, length, and metadata."""
    exit_code = main(["document", sample_text_file, "--chunk-size", "150", "--chunk-overlap", "30", "-n", "2"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "DOCUMENT INGESTION & CHUNKING REPORT" in captured.out
    assert "Total Chunks:" in captured.out
    assert "start_index:" in captured.out


def test_cli_document_missing_file_error(capsys):
    """CLI document command exits non-zero with friendly error when file is missing."""
    exit_code = main(["document", "/tmp/does_not_exist_file.txt"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: File not found" in captured.err
    assert "Traceback" not in captured.err


def test_cli_document_empty_file_error(empty_text_file, capsys):
    """CLI document command exits non-zero with friendly error when file is empty."""
    exit_code = main(["document", empty_text_file])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: Document" in captured.err
    assert "is empty or whitespace-only" in captured.err
    assert "Traceback" not in captured.err

