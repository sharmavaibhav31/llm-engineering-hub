"""Document ingestion and chunking module using LangChain.

Loads text files with TextLoader and splits them using
RecursiveCharacterTextSplitter with start_index tracking.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lcel_pipeline.config import PipelineSettings


@dataclass(frozen=True)
class SplitSummary:
    """Summary of a document splitting operation."""
    file_path: str
    total_characters: int
    total_chunks: int
    chunk_size: int
    chunk_overlap: int
    chunks: List[Document]


def load_and_split_document(
    file_path: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
) -> SplitSummary:
    """Load a text file and split it into chunks.

    Args:
        file_path: Path to the .txt file.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between adjacent chunks.

    Returns:
        SplitSummary containing total characters, chunk count, and list of Document chunks.

    Raises:
        ValueError: If file_path is missing or the document is empty/whitespace.
        FileNotFoundError: If the specified file does not exist.
    """
    if not file_path or not file_path.strip():
        raise ValueError("File path must be provided for document mode.")

    cleaned_path = file_path.strip()
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"File not found: '{cleaned_path}'. Please provide a valid file path.")

    if not os.path.isfile(cleaned_path):
        raise ValueError(f"Path is not a regular file: '{cleaned_path}'.")

    # Load document using TextLoader
    loader = TextLoader(cleaned_path, encoding="utf-8")
    docs = loader.load()

    if not docs or not docs[0].page_content.strip():
        raise ValueError(f"Document '{cleaned_path}' is empty or whitespace-only. No chunks produced.")

    total_chars = len(docs[0].page_content)

    # Split using RecursiveCharacterTextSplitter with start_index tracking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)

    return SplitSummary(
        file_path=cleaned_path,
        total_characters=total_chars,
        total_chunks=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunks=chunks,
    )


def format_document_output(summary: SplitSummary, num_chunks: int = 3) -> str:
    """Format the chunk inspection report."""
    display_count = min(num_chunks, summary.total_chunks)
    lines = [
        "\n============================================================",
        "DOCUMENT INGESTION & CHUNKING REPORT",
        "============================================================",
        f"Source File:       {summary.file_path}",
        f"Total Characters:  {summary.total_characters:,}",
        f"Total Chunks:      {summary.total_chunks}",
        f"Chunk Size / Lap:  {summary.chunk_size} / {summary.chunk_overlap}",
        f"Showing:           First {display_count} of {summary.total_chunks} chunks",
        "------------------------------------------------------------",
    ]

    for idx, chunk in enumerate(summary.chunks[:num_chunks], start=1):
        start_idx = chunk.metadata.get("start_index", "N/A")
        content_preview = chunk.page_content.strip()
        lines.append(
            f"\n[Chunk {idx}/{summary.total_chunks}] Length: {len(chunk.page_content)} chars | start_index: {start_idx}"
        )
        lines.append(f"Metadata: {chunk.metadata}")
        lines.append(f"Content:\n{content_preview}")
        lines.append("-" * 60)

    return "\n".join(lines)


def run_document_mode(
    file_path: Optional[str],
    settings: PipelineSettings,
    num_chunks: int = 3,
) -> str:
    """Execute document mode and return formatted output."""
    summary = load_and_split_document(
        file_path=file_path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return format_document_output(summary, num_chunks=num_chunks)

