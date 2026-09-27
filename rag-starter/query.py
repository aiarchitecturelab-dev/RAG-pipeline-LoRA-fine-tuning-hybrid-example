"""Ask a question against the index.

    python query.py "What's our parental leave policy?" --role employee --dry-run
    python query.py "What's our parental leave policy?" --role employee

--dry-run prints the retrieved sources with their scores and the EXACT prompt
that would be sent to the model, then stops. It needs no API key and makes no
network call, so it is the best way to see (and debug) what the LLM will see.

Without --dry-run the prompt is sent to the Anthropic API (needs
ANTHROPIC_API_KEY and `pip install anthropic`).
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from rag.config import ROLES, Config, allowed_access_levels, load_dotenv
from rag.console import make_console_safe
from rag.pipeline import PreparedQuery, RagPipeline
from rag.prompting import IDK_PHRASE
from rag.retrieve import Hit
from rag.vectorstore import IndexMismatchError

RULE = "=" * 72


def positive_int(text: str) -> int:
    """argparse type: a whole number of at least 1 (for --top-k and --top-n)."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"must be a whole number of at least 1, got {text!r}"
        ) from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question over the indexed documents.")
    parser.add_argument("question", help="the question to ask (put it in quotes)")
    parser.add_argument(
        "--role", choices=ROLES, default="employee",
        help="who is asking; decides which documents may be retrieved (default: employee)",
    )
    parser.add_argument("--index", default="index", help="index folder (default: index)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the retrieved sources and the exact prompt; do not call the LLM",
    )
    parser.add_argument(
        "--show-chunks", action="store_true",
        help="print the full text of each retrieved chunk under its score",
    )
    parser.add_argument(
        "--top-k", type=positive_int, help="override TOP_K (candidates fetched; at least 1)"
    )
    parser.add_argument(
        "--top-n", type=positive_int, help="override RERANK_TOP_N (chunks kept; at least 1)"
    )
    parser.add_argument(
        "--explain-access", action="store_true",
        help="DEBUG: also print how many chunks the permission filter removed "
             "(never show this to end users; it reveals that hidden documents exist)",
    )
    return parser.parse_args(argv)


def format_hits(hits: Sequence[Hit], show_chunks: bool) -> str:
    """One line per source (plus optionally its text), numbered like the prompt."""
    if not hits:
        return "  (no sources retrieved)"
    lines = []
    for number, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        lines.append(
            f"[{number}] score={hit.score:.3f} (vector {hit.vector_score:.3f})  "
            f"{chunk.source} | {chunk.locator} | access={chunk.access} | updated={chunk.updated}"
        )
        if show_chunks:
            lines.append("    " + chunk.text.replace("\n", "\n    "))
            lines.append("")
    return "\n".join(lines).rstrip()


def print_retrieval(prepared: PreparedQuery, args: argparse.Namespace, config: Config) -> None:
    allowed = ", ".join(sorted(allowed_access_levels(prepared.role)))
    top_k = args.top_k if args.top_k is not None else config.top_k
    top_n = args.top_n if args.top_n is not None else config.rerank_top_n
    print(f"Question : {prepared.question}")
    print(f"Role     : {prepared.role} (may read access levels: {allowed})")
    print(f"Retrieval: {prepared.candidates} candidates -> kept {len(prepared.hits)} "
          f"after re-ranking (TOP_K={top_k}, "
          f"RERANK_TOP_N={top_n}, MIN_SCORE={config.min_score})")
    if args.explain_access:
        print(f"Access   : the permission filter removed {prepared.excluded_by_permissions} "
              f"chunk(s) this role may not read [debug]")
    print()
    print(RULE)
    print("RETRIEVED SOURCES (after permission filter, similarity search and re-ranking)")
    print(RULE)
    print(format_hits(prepared.hits, args.show_chunks))
    print()


def run_dry(prepared: PreparedQuery) -> None:
    if not prepared.hits:
        print("[dry run] No allowed, relevant sources were found. A live run would NOT")
        print("call the model; it would answer:")
        print(f'  "{IDK_PHRASE}"')
        return
    print(RULE)
    print("SYSTEM PROMPT (exact text)")
    print(RULE)
    print(prepared.prompt.system)
    print()
    print(RULE)
    print("USER PROMPT (exact text)")
    print(RULE)
    print(prepared.prompt.user)
    print()
    print("[dry run] Nothing was sent to any API.")


def run_live(pipeline: RagPipeline, prepared: PreparedQuery) -> int:
    # Imported here so that --dry-run works even if the anthropic package is not installed.
    try:
        from rag.llm import LLMError
    except ImportError:
        print("error: live queries need the Anthropic SDK: pip install anthropic "
              "(or add --dry-run to see the prompt without calling the model)",
              file=sys.stderr)
        return 1

    try:
        answer = pipeline.answer(prepared)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(RULE)
    print("ANSWER")
    print(RULE)
    print(answer.text)
    print()
    if not answer.called_llm:
        print("(No allowed, relevant sources were found, so the model was not called.)")
    for warning in answer.warnings:
        print(f"WARNING: {warning}")
    report = answer.citations
    if report is not None:
        if report.unknown:
            tags = ", ".join(c.text for c in report.unknown)
            print(f"WARNING: citation(s) that do not match any retrieved source: {tags}")
        elif report.answered_without_citations:
            print("WARNING: the answer contains no citations.")
        else:
            print(f"Citation check: {len(report.cited)} citation(s), all match retrieved sources.")
    return 0


def main(argv: list[str] | None = None) -> int:
    make_console_safe()
    load_dotenv()
    args = parse_args(argv)

    try:
        config = Config.from_env()
        try:
            pipeline = RagPipeline.from_index(args.index, config)
        except (KeyError, TypeError) as exc:
            # A missing key or an unexpected field in chunks.json: the index is damaged
            # or was written by another version of this template. Say so, do not crash.
            print(
                f"error: the index in {args.index!r} looks corrupt or was written by a "
                f"different version of this template ({type(exc).__name__}: {exc}). "
                f"Rebuild it with: python ingest.py --docs data/docs --index {args.index}",
                file=sys.stderr,
            )
            return 1
        prepared = pipeline.prepare(args.question, args.role, top_k=args.top_k, top_n=args.top_n)
    except (FileNotFoundError, IndexMismatchError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_retrieval(prepared, args, config)
    if args.dry_run:
        run_dry(prepared)
        return 0
    return run_live(pipeline, prepared)


if __name__ == "__main__":
    sys.exit(main())
