"""Tests for lcel_pipeline/experiment.py."""

import os
import pytest
from lcel_pipeline.experiment import (
    NOTES_PATH,
    TOPIC_HEADINGS,
    extract_topic_spans,
)
from lcel_pipeline.document import load_and_split_document


def test_notes_file_exists_and_has_five_topics():
    """notes.txt must exist, have 5 topics, and be ~1,300-1,500 words."""
    assert os.path.exists(NOTES_PATH)
    with open(NOTES_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    words = content.split()
    assert len(words) >= 1200  # about 1,500 words

    for heading in TOPIC_HEADINGS:
        assert heading in content

    spans = extract_topic_spans(content)
    assert len(spans) == 5


def test_comparative_chunking_counts():
    """Verify 400/50 produces significantly more chunks than 1200/200."""
    res_400 = load_and_split_document(NOTES_PATH, chunk_size=400, chunk_overlap=50)
    res_1200 = load_and_split_document(NOTES_PATH, chunk_size=1200, chunk_overlap=200)

    assert res_400.total_chunks == 37
    assert res_1200.total_chunks == 14
    assert res_400.total_chunks > res_1200.total_chunks

