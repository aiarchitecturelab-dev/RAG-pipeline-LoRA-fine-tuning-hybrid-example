"""Step 4 of the pipeline: store vectors and search them WITH METADATA FILTERS.

This is a deliberately tiny stand-in for a vector database. It keeps every
vector in one numpy matrix and finds neighbors by brute force. That is fine
for thousands of chunks and easy to read; beyond that, swap it for a real
vector database (see the README) - but keep the same contract:

    search(query_vector, top_k, *, filters, min_score=None, allow_unfiltered=False)

The important design point is the `filters` argument. It is required, and the
store FAILS CLOSED: filters=None or an empty mapping raises ValueError unless the
caller says allow_unfiltered=True on purpose. A forgotten filter therefore stops
the program instead of quietly searching every chunk. Filtering happens BEFORE
similarity is computed:

    * Chunks the caller may not see are never scored, so they cannot appear in
      the results, cannot use up a top-k slot, and cannot leak through their
      scores.
    * With "search first, filter afterwards" you can end up with fewer than k
      results (or none) even though allowed chunks exist, and restricted text
      has already been touched by the search. Prefer a store that pre-filters
      natively.

On disk an index is two files in one folder:
    vectors.npz   the embedding matrix (numpy, compressed)
    chunks.json   the chunk texts and metadata, plus which embedder built it
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Collection, Mapping, Optional

import numpy as np

from .chunking import Chunk

VECTORS_FILE = "vectors.npz"
CHUNKS_FILE = "chunks.json"
FORMAT_VERSION = 1

Filters = Mapping[str, Collection[str]]


class IndexMismatchError(RuntimeError):
    """The index was built with a different embedder than the one in use."""


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0  # all-zero rows stay all-zero instead of becoming NaN
    return (matrix / norms).astype(np.float32)


class VectorStore:
    """In-memory cosine-similarity store with metadata filtering."""

    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        embedder_name: str,
        build_info: Optional[dict[str, Any]] = None,
    ) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2-D array (chunks x dimension)")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"{len(chunks)} chunks but {embeddings.shape[0]} embedding rows"
            )
        self.chunks = list(chunks)
        # Normalize once so that cosine similarity is just a dot product.
        self._embeddings = _normalise_rows(embeddings)
        self.embedder_name = embedder_name
        self.build_info = dict(build_info or {})

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def dimension(self) -> int:
        return int(self._embeddings.shape[1])

    # ------------------------------------------------------------------ search

    def _allowed_mask(self, filters: Optional[Filters]) -> np.ndarray:
        """Boolean mask: True for chunks whose metadata passes EVERY filter."""
        mask = np.ones(len(self.chunks), dtype=bool)
        for field_name, allowed_values in (filters or {}).items():
            if field_name not in Chunk.__dataclass_fields__:
                # Fail closed: silently ignoring an unknown filter would return
                # everything, which is the opposite of what the caller asked for.
                raise ValueError(f"Cannot filter on unknown chunk field {field_name!r}")
            allowed = set(allowed_values)
            mask &= np.array(
                [getattr(chunk, field_name) in allowed for chunk in self.chunks],
                dtype=bool,
            )
        return mask

    def count_matching(self, filters: Optional[Filters]) -> int:
        """How many chunks pass the filters (used to report how many were excluded).

        This only counts; it returns no chunk. With no filters every chunk counts.
        """
        return int(self._allowed_mask(filters).sum())

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        *,
        filters: Optional[Filters],
        min_score: Optional[float] = None,
        allow_unfiltered: bool = False,
    ) -> list[tuple[Chunk, float]]:
        """Return up to top_k (chunk, cosine_score) pairs, best first.

        `filters` is required (keyword-only). It maps a chunk field to the
        values that are allowed, e.g. {"access": {"all", "managers"}}. A chunk
        must pass every filter. An empty allowed set matches nothing.

        Fail closed: filters=None or an empty mapping raises ValueError, because
        it would search every chunk, restricted ones included. Pass
        allow_unfiltered=True only when you really mean it (a test, an admin
        tool), never for a request that comes from a user.
        """
        if not filters and not allow_unfiltered:
            raise ValueError(
                "search() needs non-empty filters (for example "
                "{'access': {'all'}}); searching without a filter would also return "
                "restricted chunks. Pass allow_unfiltered=True only if that is intended."
            )
        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape != (self.dimension,):
            raise ValueError(
                f"Query vector has shape {query.shape}, expected ({self.dimension},)"
            )
        norm = float(np.linalg.norm(query))
        if norm > 0.0:
            query = query / norm

        candidate_rows = np.flatnonzero(self._allowed_mask(filters))  # FILTER FIRST
        if candidate_rows.size == 0 or top_k < 1:
            return []

        scores = self._embeddings[candidate_rows] @ query  # then similarity
        # Stable sort: equal scores keep insertion order, so results are deterministic.
        order = np.argsort(-scores, kind="stable")[:top_k]  # then top-k
        results = []
        for position in order:
            score = float(scores[position])
            if min_score is not None and score < min_score:
                break  # scores are sorted, so everything after this is lower too
            results.append((self.chunks[int(candidate_rows[position])], score))
        return results

    # ------------------------------------------------------------ persistence

    def save(self, directory: str | Path) -> None:
        """Write vectors.npz and chunks.json into `directory` (created if needed)."""
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / VECTORS_FILE, embeddings=self._embeddings)
        payload = {
            "format_version": FORMAT_VERSION,
            "embedder": self.embedder_name,
            "dimension": self.dimension,
            "build_info": self.build_info,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
        (folder / CHUNKS_FILE).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """Read an index written by save()."""
        folder = Path(directory)
        vectors_path = folder / VECTORS_FILE
        chunks_path = folder / CHUNKS_FILE
        if not vectors_path.is_file() or not chunks_path.is_file():
            raise FileNotFoundError(
                f"No index found in {folder}. Run this from inside the rag-starter folder. "
                "Build one first, for example:\n"
                f"    python ingest.py --docs data/docs --index {folder}"
            )
        try:
            payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError are both ValueErrors
            raise ValueError(
                f"{chunks_path} is not valid JSON ({exc}); the index looks corrupt. "
                "Re-run ingest.py to rebuild it."
            ) from None
        if not isinstance(payload, dict):
            raise ValueError(
                f"{chunks_path} does not look like an index file; re-run ingest.py to rebuild it."
            )
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported index format {payload.get('format_version')!r} in {folder}; "
                "re-run ingest.py."
            )
        # allow_pickle=False: never execute code hidden inside a data file.
        with np.load(vectors_path, allow_pickle=False) as archive:
            embeddings = archive["embeddings"]
        chunks = [Chunk.from_dict(item) for item in payload["chunks"]]
        return cls(chunks, embeddings, payload["embedder"], payload.get("build_info"))

    def check_embedder(self, embedder_name: str, dimension: int) -> None:
        """Raise IndexMismatchError unless the embedder matches the one that built this index.

        Vectors from two different models live in different spaces: comparing
        them gives meaningless scores, not an error. So we check explicitly.
        """
        if embedder_name != self.embedder_name or dimension != self.dimension:
            raise IndexMismatchError(
                f"This index was built with embedder {self.embedder_name!r} "
                f"(dimension {self.dimension}) but you are querying with "
                f"{embedder_name!r} (dimension {dimension}). Set EMBEDDER to match, "
                "or run ingest.py again to rebuild the index."
            )
