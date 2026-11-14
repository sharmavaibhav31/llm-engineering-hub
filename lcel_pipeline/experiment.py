"""Chunk boundary experiment comparing 400/50 vs 1200/200 on notes.txt.

Demonstrates chunk count differences, boundary cuts across topics,
and names specific technical concepts split across chunk borders.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple
from langchain_core.documents import Document

from lcel_pipeline.document import load_and_split_document

NOTES_PATH = os.path.join(os.path.dirname(__file__), "notes.txt")

TOPIC_HEADINGS = [
    "## Topic 1: Vector Embeddings and Semantic Metric Spaces",
    "## Topic 2: Hierarchical Navigable Small World (HNSW) Indexing",
    "## Topic 3: Context Dilution and Chunking Dynamics",
    "## Topic 4: Hybrid Search and Reciprocal Rank Fusion (RRF)",
    "## Topic 5: Cross-Encoder Re-Ranking and Lossless Compression",
]


def extract_topic_spans(raw_text: str) -> List[Tuple[str, int, int]]:
    """Extract character start and end indices for each topic in notes.txt."""
    spans = []
    positions = []
    for heading in TOPIC_HEADINGS:
        pos = raw_text.find(heading)
        if pos != -1:
            positions.append((heading, pos))

    positions.sort(key=lambda x: x[1])

    for i in range(len(positions)):
        name, start = positions[i]
        end = positions[i + 1][1] if i + 1 < len(positions) else len(raw_text)
        spans.append((name, start, end))

    return spans


def analyze_setting(
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
    raw_text: str,
    topic_spans: List[Tuple[str, int, int]],
) -> None:
    """Analyze chunking results for a specific chunk_size / chunk_overlap setting."""
    summary = load_and_split_document(
        file_path=file_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    print("\n" + "=" * 70)
    print(f"EXPERIMENT SETTING: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print("=" * 70)
    print(f"Total Document Characters: {summary.total_characters:,}")
    print(f"Total Chunks Generated:    {summary.total_chunks}")
    avg_len = sum(len(c.page_content) for c in summary.chunks) / summary.total_chunks
    print(f"Average Chunk Length:      {avg_len:.1f} characters")

    # Map chunks to topics
    topic_chunk_map: Dict[str, List[int]] = {name: [] for name, _, _ in topic_spans}
    boundary_crossings: List[Tuple[int, str, str]] = []

    for idx, chunk in enumerate(summary.chunks, start=1):
        start = chunk.metadata.get("start_index", 0)
        end = start + len(chunk.page_content)

        matching_topics = []
        for name, t_start, t_end in topic_spans:
            # Check for overlap between [start, end) and [t_start, t_end)
            if max(start, t_start) < min(end, t_end):
                matching_topics.append(name)
                topic_chunk_map[name].append(idx)

        if len(matching_topics) > 1:
            boundary_crossings.append((idx, matching_topics[0], matching_topics[1]))

    print("\n--- Topic Coverage & Chunk Allocation ---")
    for name, _, _ in topic_spans:
        short_name = name.replace("## ", "").split(":")[0] + ": " + name.split(":")[1].strip()
        assigned_chunks = topic_chunk_map[name]
        print(f"  * {short_name:<40} -> {len(assigned_chunks)} chunks (IDs: {assigned_chunks[0]} to {assigned_chunks[-1]})")

    print("\n--- Cross-Topic Boundary Crossings ---")
    if boundary_crossings:
        print(f"Found {len(boundary_crossings)} chunk(s) straddling multiple topic boundaries:")
        for chunk_id, t_prev, t_next in boundary_crossings:
            prev_label = t_prev.split(":")[0].replace("## ", "")
            next_label = t_next.split(":")[0].replace("## ", "")
            print(f"  * Chunk #{chunk_id} straddles [{prev_label}] and [{next_label}]")
    else:
        print("  None. Clean paragraph breaks prevented chunks from mixing different top-level topics.")

    # Named concepts split analysis
    print("\n--- Named Concept Boundary Cuts ---")
    inspect_named_concept_cuts(summary.chunks, chunk_size)


def inspect_named_concept_cuts(chunks: List[Document], chunk_size: int) -> None:
    """Identify concrete technical concepts severed across adjacent chunk boundaries."""
    key_concepts = [
        ("SIMD matrix multiplications (AVX-512/ARM Neon)", r"AVX-512|ARM Neon|SIMD inner product"),
        ("HNSW Graph Traversal & M/efConstruction", r"efConstruction|bidirectional connection links|Delaunay-like"),
        ("RRF Mathematical Formula", r"RRF_Score|smoothing hyperparameter|Okapi BM25"),
        ("Cross-Encoder Complexity O(N * (L_q + L_d)^2)", r"L_q \+ L_d|cross-encoders require full forward-pass"),
        ("Distributional Hypothesis & InfoNCE", r"InfoNCE|distributional hypothesis"),
    ]

    for concept_name, pattern in key_concepts:
        found_in_chunks = []
        for idx, chunk in enumerate(chunks, start=1):
            if re.search(pattern, chunk.page_content):
                found_in_chunks.append(idx)

        if len(found_in_chunks) > 1:
            print(f"  [SPLIT ACROSS CHUNKS]: '{concept_name}'")
            print(f"    Spans across chunks: {found_in_chunks}")
            c1, c2 = chunks[found_in_chunks[0] - 1], chunks[found_in_chunks[1] - 1]
            print(f"    Chunk #{found_in_chunks[0]} tail: ...{c1.page_content[-70:].strip()!r}")
            print(f"    Chunk #{found_in_chunks[1]} head: {c2.page_content[:70].strip()!r}...")
        elif len(found_in_chunks) == 1:
            print(f"  [CONTAINED WHOLE]:     '{concept_name}' (fit completely inside Chunk #{found_in_chunks[0]})")


def run_experiment() -> None:
    """Execute the comparative chunking experiment."""
    if not os.path.exists(NOTES_PATH):
        raise FileNotFoundError(f"Notes file not found at: {NOTES_PATH}")

    with open(NOTES_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    topic_spans = extract_topic_spans(raw_text)

    print("\n======================================================================")
    print("      RAG CHUNKING EXPERIMENT: notes.txt (1,343 words, 5 topics)      ")
    print("======================================================================")

    # Setting 1: 400 / 50
    analyze_setting(
        file_path=NOTES_PATH,
        chunk_size=400,
        chunk_overlap=50,
        raw_text=raw_text,
        topic_spans=topic_spans,
    )

    # Setting 2: 1200 / 200
    analyze_setting(
        file_path=NOTES_PATH,
        chunk_size=1200,
        chunk_overlap=200,
        raw_text=raw_text,
        topic_spans=topic_spans,
    )


if __name__ == "__main__":
    run_experiment()

