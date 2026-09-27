"""Ingest documents into a searchable index.

    python ingest.py --docs data/docs --index index

Steps (pipeline steps 1-4): load documents -> chunk -> embed -> save the index.
Run it again whenever a document changes. It uses the offline hashing embedder
unless EMBEDDER says otherwise, so no network is needed by default.

Exit codes: 0 = index built (warnings, if any, are printed to stderr);
1 = nothing was built (a bad or duplicated front matter key, a missing access
level, no documents, or documents that produce no chunks at all).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from rag.config import Config, load_dotenv
from rag.console import make_console_safe
from rag.pipeline import build_index
from rag.vectorstore import CHUNKS_FILE, VECTORS_FILE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the RAG index from a folder of documents."
    )
    parser.add_argument(
        "--docs", default="data/docs",
        help="folder with .md/.txt/.pdf files (default: data/docs)",
    )
    parser.add_argument(
        "--index", default="index", help="output folder for the index (default: index)"
    )
    parser.add_argument("--chunk-words", type=int, help="override CHUNK_WORDS")
    parser.add_argument(
        "--chunk-overlap", type=int,
        help="override CHUNK_OVERLAP (default with --chunk-words: min(CHUNK_OVERLAP, chunk_words // 4))",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    make_console_safe()
    load_dotenv()
    args = parse_args(argv)

    try:
        config = Config.from_env()
        overrides = {}
        if args.chunk_words is not None:
            overrides["chunk_words"] = args.chunk_words
            # Keep the overlap sensible for small chunks unless it was set explicitly.
            overrides["chunk_overlap"] = min(config.chunk_overlap, args.chunk_words // 4)
        if args.chunk_overlap is not None:
            overrides["chunk_overlap"] = args.chunk_overlap
        config = dataclasses.replace(config, **overrides)  # re-validates the values
        summary = build_index(args.docs, args.index, config)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {summary.documents} documents from {args.docs}")
    print(
        f"Created {summary.chunks} chunks "
        f"(chunk_words={config.chunk_words}, overlap={config.chunk_overlap})"
    )
    for access, count in sorted(summary.chunks_by_access.items()):
        print(f"  access={access:<10} {count} chunks")
    print(f"Embedder: {summary.embedder} (dimension {summary.dimension})")
    print(f"Index saved to {args.index} ({VECTORS_FILE}, {CHUNKS_FILE})")
    # Problems that do not stop the run but that a human should read: chunks that
    # can never be retrieved (for example non-English text with the hashing
    # embedder) and documents that produced no chunks. They go to stderr.
    for warning in summary.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
