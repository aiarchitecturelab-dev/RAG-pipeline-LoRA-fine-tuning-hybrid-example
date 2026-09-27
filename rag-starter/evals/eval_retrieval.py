"""Measure retrieval quality on a small labeled question set.

    python evals/eval_retrieval.py
    python evals/eval_retrieval.py --chunk-words 60
    python evals/eval_retrieval.py --sweep 30,60,120,220,400
    python evals/eval_retrieval.py --no-rerank --verbose

What it measures (and nothing else): for each labeled question it runs the real
retrieval pipeline (permission filter -> similarity -> optional re-rank) and
checks where the expected passage shows up in the ranked results.

    section hit@k   a chunk from the expected file AND the expected section is
                    within the top k results
    evidence hit@k  same, and that chunk's text also contains the answer phrase
                    (the "evidence" column in data/eval_questions.jsonl)

Section hit tells you whether retrieval found the right place; evidence hit
tells you whether the chunk actually contains the answer. When chunks are made
smaller than their section, the two can drift apart - which is exactly the
chunk-size effect you can watch with --chunk-words or --sweep.

Questions with "expected_source": null are ACCESS-CONTROL PROBES. They are not
scored as hits. Instead the script checks that no result, for any question,
carries an access level the asking role may not read. That check runs in every
mode: the default report prints the total, and --sweep prints it for every row.
Exit codes: 0 = done and no restricted chunk was returned; 1 = bad input or
setup; 2 = at least one restricted chunk was returned (a permission bug).

Caveats: 18 questions on 4 tiny documents is a demonstration, not a benchmark.
The numbers say nothing about your own documents - build your own question set.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

# Make "import rag" work when this file is run as "python evals/eval_retrieval.py".
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag.config import Config, allowed_access_levels, load_dotenv  # noqa: E402
from rag.console import make_console_safe  # noqa: E402
from rag.embedders import Embedder, get_embedder  # noqa: E402
from rag.loaders import load_documents  # noqa: E402
from rag.pipeline import RagPipeline, build_store  # noqa: E402
from rag.rerank import (  # noqa: E402
    CrossEncoderReranker,
    LexicalReranker,
    PassthroughReranker,
    Reranker,
)
from rag.retrieve import Hit  # noqa: E402

DEFAULT_KS = (1, 3, 5)
RERANKERS = ("lexical", "none", "cross-encoder")


def make_reranker(name: str) -> Reranker:
    """Create the re-ranker selected on the command line."""
    if name == "lexical":
        return LexicalReranker()
    if name == "none":
        return PassthroughReranker()
    if name == "cross-encoder":
        return CrossEncoderReranker()  # optional: needs sentence-transformers
    raise ValueError(f"Unknown re-ranker {name!r}. Use one of: {', '.join(RERANKERS)}")


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc})") from None
    return questions


def _squash(text: str) -> str:
    return " ".join(text.split()).casefold()


def first_rank(hits: Sequence[Hit], question: dict[str, Any], need_evidence: bool) -> Optional[int]:
    """1-based rank of the first matching hit, or None."""
    expected_source = question["expected_source"]
    expected_section = _squash(question["expected_section"])
    evidence = _squash(question["evidence"]) if need_evidence else None
    for rank, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        if chunk.source != expected_source or _squash(chunk.section) != expected_section:
            continue
        if evidence is not None and evidence not in _squash(chunk.text):
            continue
        return rank
    return None


@dataclasses.dataclass
class EvalResult:
    chunks: int
    answerable: int
    probes: int
    section_ranks: list[Optional[int]]
    evidence_ranks: list[Optional[int]]
    access_violations: int
    details: list[dict[str, Any]]

    def hit_at(self, ranks: list[Optional[int]], k: int) -> int:
        return sum(1 for rank in ranks if rank is not None and rank <= k)


def run_eval(pipeline: RagPipeline, questions: list[dict[str, Any]], ks: Sequence[int]) -> EvalResult:
    max_k = max(ks)
    section_ranks: list[Optional[int]] = []
    evidence_ranks: list[Optional[int]] = []
    details: list[dict[str, Any]] = []
    violations = 0
    probes = 0

    for question in questions:
        prepared = pipeline.prepare(question["question"], question["role"], top_n=max_k)
        allowed = allowed_access_levels(question["role"])
        violations += sum(1 for hit in prepared.hits if hit.chunk.access not in allowed)

        if question["expected_source"] is None:
            probes += 1
            details.append({"id": question["id"], "probe": True, "returned": len(prepared.hits)})
            continue

        section_rank = first_rank(prepared.hits, question, need_evidence=False)
        evidence_rank = first_rank(prepared.hits, question, need_evidence=True)
        section_ranks.append(section_rank)
        evidence_ranks.append(evidence_rank)
        details.append(
            {"id": question["id"], "probe": False, "question": question["question"],
             "section_rank": section_rank, "evidence_rank": evidence_rank}
        )

    return EvalResult(
        chunks=len(pipeline.store),
        answerable=len(section_ranks),
        probes=probes,
        section_ranks=section_ranks,
        evidence_ranks=evidence_ranks,
        access_violations=violations,
        details=details,
    )


def build_pipeline(
    docs_dir: Path, config: Config, embedder: Embedder, reranker: Reranker
) -> RagPipeline:
    """Chunk and embed the documents in memory (nothing is written to disk)."""
    store = build_store(load_documents(docs_dir), embedder, config)
    return RagPipeline(store, embedder, config, reranker)


def _pct(count: int, total: int) -> str:
    return f"{count}/{total} ({100.0 * count / total:5.1f}%)" if total else "n/a"


def print_report(result: EvalResult, config: Config, embedder: Embedder, reranker_name: str,
                 ks: Sequence[int], verbose: bool) -> None:
    print("Retrieval eval")
    print(f"  questions : {result.answerable} answerable + {result.probes} access-control probes")
    print(f"  chunking  : chunk_words={config.chunk_words} overlap={config.chunk_overlap} "
          f"-> {result.chunks} chunks in the index")
    print(f"  embedder  : {embedder.name}   re-rank: {_rerank_label(reranker_name)}   "
          f"top_k={config.top_k}")
    print()
    for label, ranks in (("section hit", result.section_ranks),
                         ("evidence hit", result.evidence_ranks)):
        cells = "   ".join(f"@{k}: {_pct(result.hit_at(ranks, k), result.answerable)}" for k in ks)
        print(f"  {label:<13} {cells}")
    print()
    print(f"  access-control check: {result.access_violations} restricted chunk(s) returned "
          f"across all {result.answerable + result.probes} queries (must be 0)")
    if verbose:
        print()
        for item in result.details:
            if item["probe"]:
                print(f"  {item['id']}  probe: {item['returned']} chunk(s) returned")
            else:
                print(f"  {item['id']}  section rank={item['section_rank']}  "
                      f"evidence rank={item['evidence_rank']}  {item['question']}")


def _rerank_label(reranker_name: str) -> str:
    return "off" if reranker_name == "none" else f"on ({reranker_name})"


def print_sweep(rows: list[tuple[int, EvalResult]], ks: Sequence[int]) -> None:
    print("Chunk-size sweep (section hit / evidence hit, counts of answerable questions)")
    header = f"  {'chunk_words':>11} {'chunks':>7}  " + "  ".join(
        f"{'sec@' + str(k):>7} {'evid@' + str(k):>7}" for k in ks
    ) + f"  {'access_viol':>11}"
    print(header)
    for chunk_words, result in rows:
        cells = "  ".join(
            f"{result.hit_at(result.section_ranks, k):>7} {result.hit_at(result.evidence_ranks, k):>7}"
            for k in ks
        )
        print(f"  {chunk_words:>11} {result.chunks:>7}  {cells}  {result.access_violations:>11}")
    total = rows[0][1].answerable if rows else 0
    print(f"  (out of {total} answerable questions; overlap = min(CHUNK_OVERLAP, chunk_words // 4))")
    print("  (access_viol = restricted chunks returned for that chunk size; must be 0)")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval hit@k.")
    parser.add_argument("--docs", default=str(PROJECT_ROOT / "data" / "docs"))
    parser.add_argument("--questions", default=str(PROJECT_ROOT / "data" / "eval_questions.jsonl"))
    parser.add_argument("--chunk-words", type=int, help="chunk size in words (default: CHUNK_WORDS)")
    parser.add_argument("--chunk-overlap", type=int,
                        help="overlap in words (default: min(CHUNK_OVERLAP, chunk_words // 4))")
    parser.add_argument("--sweep", help="comma-separated chunk sizes to compare, e.g. 30,60,120,220")
    parser.add_argument("--top-k", type=int, help="candidates fetched before re-ranking")
    parser.add_argument("--embedder", help="hashing or sentence-transformers (default: EMBEDDER)")
    parser.add_argument("--reranker", choices=RERANKERS, default="lexical",
                        help="re-ranking step (default: lexical; cross-encoder needs sentence-transformers)")
    parser.add_argument("--no-rerank", action="store_true", help="shortcut for --reranker none")
    parser.add_argument("--ks", default=",".join(map(str, DEFAULT_KS)),
                        help="values of k for hit@k (default: 1,3,5)")
    parser.add_argument("--verbose", action="store_true", help="print the rank found for every question")
    return parser.parse_args(argv)


def config_for(base: Config, chunk_words: Optional[int], overlap: Optional[int],
               top_k: Optional[int], embedder: Optional[str]) -> Config:
    # "is not None" instead of "or": an explicit 0 must reach Config validation and be
    # rejected there, not be silently replaced by the default.
    words = chunk_words if chunk_words is not None else base.chunk_words
    if overlap is None:
        overlap = min(base.chunk_overlap, words // 4)
    return dataclasses.replace(
        base,
        chunk_words=words,
        chunk_overlap=overlap,
        top_k=top_k if top_k is not None else base.top_k,
        embedder=(embedder if embedder is not None else base.embedder).lower(),
    )


def main(argv: Optional[list[str]] = None) -> int:
    make_console_safe()
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args(argv)
    try:
        ks = tuple(sorted({int(part) for part in args.ks.split(",") if part.strip()}))
        base = Config.from_env()
        questions = load_questions(Path(args.questions))
        reranker_name = "none" if args.no_rerank else args.reranker
        sizes = [int(part) for part in args.sweep.split(",")] if args.sweep else None

        first = config_for(base, (sizes or [args.chunk_words])[0], args.chunk_overlap,
                           args.top_k, args.embedder)
        embedder = get_embedder(first.embedder, first.sentence_model)  # load the model once
        reranker = make_reranker(reranker_name)

        if sizes:
            rows = []
            for size in sizes:
                config = config_for(base, size, args.chunk_overlap, args.top_k, args.embedder)
                pipeline = build_pipeline(Path(args.docs), config, embedder, reranker)
                rows.append((size, run_eval(pipeline, questions, ks)))
            print_sweep(rows, ks)
            print(f"  embedder={embedder.name}  re-rank={_rerank_label(reranker_name)}  "
                  f"top_k={first.top_k}")
            violations = sum(result.access_violations for _, result in rows)
            print(f"  access-control check: {violations} restricted chunk(s) returned "
                  f"across all {len(rows)} sweep row(s) (must be 0)")
            if violations:
                print("  FAILED: a restricted chunk was returned; exiting with code 2.")
            return 0 if violations == 0 else 2

        pipeline = build_pipeline(Path(args.docs), first, embedder, reranker)
        result = run_eval(pipeline, questions, ks)
        print_report(result, first, embedder, reranker_name, ks, args.verbose)
        return 0 if result.access_violations == 0 else 2
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
