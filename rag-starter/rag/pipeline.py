"""The glue: wire loaders, chunker, embedder, store, retriever, re-ranker,
prompt builder and LLM into two operations.

    build_index(...)          ingestion: documents -> chunks -> vectors -> index on disk
    RagPipeline.prepare(...)  everything up to (but not including) the LLM call
    RagPipeline.answer(...)   the LLM call plus the citation check

Splitting "prepare" from "answer" is what makes --dry-run possible: you can
look at exactly what would be sent to the model without sending anything.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .chunking import Chunk, chunk_documents
from .config import Config
from .embedders import Embedder, HashingEmbedder, get_embedder
from .loaders import Document, load_documents
from .prompting import IDK_PHRASE, CitationReport, Prompt, build_prompt, validate_citations
from .rerank import LexicalReranker, Reranker
from .retrieve import Hit, retrieve
from .text import tokenize
from .vectorstore import VectorStore


# ---------------------------------------------------------------- ingestion


@dataclass(frozen=True)
class IngestSummary:
    documents: int
    chunks: int
    chunks_by_access: dict[str, int]
    embedder: str
    dimension: int
    warnings: tuple[str, ...] = ()   # things a human should look at (printed by ingest.py)


# How many individual chunk warnings to list before summarizing the rest.
MAX_LISTED_WARNINGS = 10


def ingest_warnings(
    documents: Sequence[Document],
    chunks: Sequence[Chunk],
    vectors: np.ndarray,
    embedder: Embedder,
) -> list[str]:
    """Describe documents and chunks that a search will not find by their content.

    Three problems are reported, each by file (and chunk) name:
      * a document that produced no chunks at all (empty file, only front
        matter, or a scanned PDF without a text layer);
      * a chunk whose vector is all zero. A zero vector has cosine similarity 0
        with everything, so that chunk can never be retrieved;
      * with the default hashing embedder, a chunk whose TEXT has no tokens even
        though its heading does: it is then indexed by its heading alone.
    The hashing embedder (via rag/text.py) only sees ASCII letters and digits, so
    all of this happens silently for Nepali, Chinese, Arabic or other non-ASCII text.
    """
    warnings: list[str] = []

    chunk_counts = Counter(chunk.source for chunk in chunks)
    for document in documents:
        if chunk_counts[document.source] == 0:
            warnings.append(
                f"{document.source}: no chunks were created from this document (empty file, "
                "only front matter, or a PDF without a text layer?). It cannot be searched."
            )

    zero_rows = set(np.flatnonzero(np.linalg.norm(vectors, axis=1) == 0.0).tolist())
    lexical = isinstance(embedder, HashingEmbedder)  # its tokens come from rag/text.py
    flagged: list[str] = []
    for row, chunk in enumerate(chunks):
        where = f" (section {chunk.section!a})" if chunk.section else ""
        if row in zero_rows:
            flagged.append(
                f"{chunk.source}, chunk {chunk.chunk_id}{where}: no searchable text, so its "
                "vector is all zero and it can never be retrieved."
            )
        elif lexical and not tokenize(chunk.text):
            flagged.append(
                f"{chunk.source}, chunk {chunk.chunk_id}{where}: its text has no ASCII letters "
                "or digits, so it is indexed by its section heading only."
            )
    warnings.extend(flagged[:MAX_LISTED_WARNINGS])
    if len(flagged) > MAX_LISTED_WARNINGS:
        warnings.append(f"... and {len(flagged) - MAX_LISTED_WARNINGS} more chunk(s) like this.")
    if flagged:
        hint = (
            "The default hashing embedder only reads ASCII letters and digits (it is "
            "English/ASCII-oriented). For other languages set EMBEDDER=sentence-transformers "
            "with a multilingual SENTENCE_MODEL (see the README, Known limitations) and "
            "ingest again."
            if lexical
            else "Check that the embedder can read this text."
        )
        warnings.append(
            f"{len(flagged)} of {len(chunks)} chunk(s) cannot be found by their text with "
            f"embedder {embedder.name!r}. {hint}"
        )
    return warnings


def build_store(
    documents: Sequence[Document],
    embedder: Embedder,
    config: Config,
    warnings: Optional[list[str]] = None,
) -> VectorStore:
    """Chunk and embed documents into an in-memory VectorStore.

    Raises ValueError when the documents yield no chunk at all: an empty index
    is never what anybody wants, and silently saving one hides the mistake.
    If a list is passed as `warnings`, problems that do not stop the run (see
    ingest_warnings) are appended to it.
    """
    chunks: list[Chunk] = chunk_documents(
        list(documents), config.chunk_words, config.chunk_overlap
    )
    if not chunks:
        raise ValueError(
            f"No chunks were created from {len(documents)} document(s): they are empty, "
            "contain only front matter, or (for PDFs) have no text layer. Nothing was indexed."
        )
    vectors = embedder.embed_documents([chunk.embed_text for chunk in chunks])
    if warnings is not None:
        warnings.extend(ingest_warnings(documents, chunks, vectors, embedder))
    return VectorStore(
        chunks,
        vectors,
        embedder_name=embedder.name,
        build_info={
            "chunk_words": config.chunk_words,
            "chunk_overlap": config.chunk_overlap,
            "documents": len(documents),
        },
    )


def build_index(
    docs_dir: str | Path,
    index_dir: str | Path,
    config: Config,
    embedder: Optional[Embedder] = None,
) -> IngestSummary:
    """Ingest a folder of documents and save the index to disk."""
    documents = load_documents(docs_dir)
    embedder = embedder or get_embedder(config.embedder, config.sentence_model)
    warnings: list[str] = []
    store = build_store(documents, embedder, config, warnings)
    store.save(index_dir)
    return IngestSummary(
        documents=len(documents),
        chunks=len(store),
        chunks_by_access=dict(Counter(chunk.access for chunk in store.chunks)),
        embedder=embedder.name,
        dimension=store.dimension,
        warnings=tuple(warnings),
    )


# ------------------------------------------------------------------ querying


@dataclass(frozen=True)
class PreparedQuery:
    """Everything known before the LLM is called."""

    question: str
    role: str
    hits: list[Hit]                 # final chunks that go into the prompt
    candidates: int                 # how many candidates retrieval returned
    excluded_by_permissions: int    # chunks hidden from this role (debug only)
    prompt: Prompt


@dataclass
class RagAnswer:
    """The outcome of a full query."""

    prepared: PreparedQuery
    text: str
    called_llm: bool
    citations: Optional[CitationReport] = None
    warnings: list[str] = field(default_factory=list)


class RagPipeline:
    """Retrieval-augmented question answering over one index."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        config: Config,
        reranker: Optional[Reranker] = None,
    ) -> None:
        store.check_embedder(embedder.name, embedder.dimension)
        self.store = store
        self.embedder = embedder
        self.config = config
        self.reranker = reranker or LexicalReranker()

    @classmethod
    def from_index(
        cls,
        index_dir: str | Path,
        config: Config,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
    ) -> "RagPipeline":
        store = VectorStore.load(index_dir)
        embedder = embedder or get_embedder(config.embedder, config.sentence_model)
        return cls(store, embedder, config, reranker)

    def prepare(
        self,
        question: str,
        role: str,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
    ) -> PreparedQuery:
        """Retrieve -> re-rank -> keep the best top_n -> build the prompt.

        top_k and top_n default to the configured values only when they are None;
        a value that is given must be at least 1 (0 is an error, not "use the default").
        """
        top_k = top_k if top_k is not None else self.config.top_k
        top_n = top_n if top_n is not None else self.config.rerank_top_n
        if top_k < 1 or top_n < 1:
            raise ValueError(f"top_k and top_n must be at least 1 (got {top_k} and {top_n}).")

        retrieval = retrieve(
            question, self.store, self.embedder, role, top_k, self.config.min_score
        )
        reranked = self.reranker.rerank(question, retrieval.hits)
        final_hits = reranked[:top_n]
        return PreparedQuery(
            question=question,
            role=role,
            hits=final_hits,
            candidates=len(retrieval.hits),
            excluded_by_permissions=retrieval.excluded_by_permissions,
            prompt=build_prompt(question, final_hits),
        )

    def answer(self, prepared: PreparedQuery, client: Optional[Any] = None) -> RagAnswer:
        """Call the LLM (unless there is nothing to answer from) and check citations."""
        if not prepared.hits:
            # No allowed, relevant source: do not spend an LLM call and do not
            # give the model a chance to answer from memory.
            return RagAnswer(prepared=prepared, text=IDK_PHRASE, called_llm=False)

        from .llm import generate_answer  # imported here so dry runs need no SDK

        result = generate_answer(
            prepared.prompt.system,
            prepared.prompt.user,
            client=client,
            model=self.config.llm_model,
        )
        # A refusal message is ours, not the model's answer, so it has no citations to check.
        refused = result.stop_reason == "refusal"
        report = None if refused else validate_citations(result.text, prepared.hits)
        return RagAnswer(
            prepared=prepared,
            text=result.text,
            called_llm=True,
            citations=report,
            warnings=list(result.warnings),
        )

    def ask(self, question: str, role: str, client: Optional[Any] = None) -> RagAnswer:
        """Convenience: prepare() then answer()."""
        return self.answer(self.prepare(question, role), client=client)
