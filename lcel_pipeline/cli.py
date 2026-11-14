"""Command Line Interface for LCEL Pipeline."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from lcel_pipeline.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL,
    load_settings,
    validate_chunk_settings,
)


def build_parser() -> argparse.ArgumentParser:
    """Build two-mode CLI parser with 'explain' and 'document' subcommands."""
    # Shared settings parser allowing flags to be passed before or after subcommands
    shared_parser = argparse.ArgumentParser(add_help=False)
    shared_parser.add_argument(
        "-m", "--model",
        default=None,
        help=f"Model identifier override (default: {DEFAULT_MODEL} or EXPLAINER_MODEL env var).",
    )
    shared_parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Character chunk size for document splitting (default: {DEFAULT_CHUNK_SIZE}).",
    )
    shared_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Character chunk overlap for document splitting (default: {DEFAULT_CHUNK_OVERLAP}).",
    )

    parser = argparse.ArgumentParser(
        prog="lcel-pipeline",
        description="LCEL Pipeline: LangChain Expression Language CLI for explanation and document chunking.",
        parents=[shared_parser],
    )

    subparsers = parser.add_subparsers(
        dest="subcommand",
        title="Subcommands",
        description="Available pipeline modes",
        required=True,
    )

    # Subcommand: explain
    explain_parser = subparsers.add_parser(
        "explain",
        help="Generate an explanation using an LCEL chain (topic, audience, format).",
        parents=[shared_parser],
    )
    explain_parser.add_argument(
        "topic",
        nargs="?",
        default=None,
        help="The technical topic to explain.",
    )
    explain_parser.add_argument(
        "-a", "--audience",
        default="software engineers",
        help="Target audience for explanation (e.g. beginner, software engineers, executive).",
    )
    explain_parser.add_argument(
        "-f", "--format",
        default="concise summary with bullet points",
        help="Desired response format (e.g. bullet points, paragraph, tutorial).",
    )
    explain_parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the formatted prompt messages without making an API call.",
    )

    # Subcommand: document
    doc_parser = subparsers.add_parser(
        "document",
        help="Ingest and chunk text files using TextLoader and RecursiveCharacterTextSplitter.",
        parents=[shared_parser],
    )
    doc_parser.add_argument(
        "file_path",
        nargs="?",
        default=None,
        help="Path to the .txt file to load and split.",
    )
    doc_parser.add_argument(
        "-n", "--num-chunks",
        type=int,
        default=3,
        help="Number of initial chunks to display (default: 3).",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate chunk settings immediately
    validate_chunk_settings(args.chunk_size, args.chunk_overlap)

    # Load and validate settings (enforcing GROQ_API_KEY)
    settings = load_settings(
        model_override=args.model,
        chunk_size_override=args.chunk_size,
        chunk_overlap_override=args.chunk_overlap,
        require_api_key=True,
    )

    if args.subcommand == "explain":
        from lcel_pipeline.explain import run_explain
        from lcel_pipeline.errors import handle_cli_error
        try:
            output = run_explain(
                topic=args.topic,
                audience=args.audience,
                format_spec=args.format,
                settings=settings,
                show_prompt=args.show_prompt,
            )
            sys.stdout.write(f"\n{output}\n")
            sys.stdout.flush()
            return 0
        except ValueError as err:
            sys.stderr.write(f"Error: {err}\n")
            return 1
        except Exception as exc:
            return handle_cli_error(exc)

    elif args.subcommand == "document":
        from lcel_pipeline.document import run_document_mode
        try:
            output = run_document_mode(
                file_path=args.file_path,
                settings=settings,
                num_chunks=args.num_chunks,
            )
            sys.stdout.write(f"{output}\n")
            sys.stdout.flush()
            return 0
        except (FileNotFoundError, ValueError) as err:
            sys.stderr.write(f"Error: {err}\n")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
