"""Step 5 of the pipeline: retrieve candidate chunks for a question.

Order of operations (this order is the whole point):

    1. embed the question
    2. PERMISSION FILTER: turn the caller's role into a metadata filter
    3. similarity search, only over chunks that passed the filter
    4. take the top-k

Access control belongs HERE, before the LLM ever sees a chunk. Do not rely on
the prompt ("do not reveal executive pay") to protect data: an LLM cannot
un-see text it was given, and prompts can be bypassed. If restricted text never
enters the prompt, it cannot leak, whatever the user types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .chunking import Chunk
from .config import allowed_access_levels
from .embedders import Embedder
from .vectorstore import VectorStore


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk with its scores.

    `vector_score` is the cosine similarity from the store and never changes.
    `score` starts equal to it; a re-ranker may replace it with a new score.
    """

    chunk: Chunk
    vector_score: float
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[Hit]
    excluded_by_permissions: int  # chunks the role's filter removed (debug information)


def permission_filter(role: str) -> dict[str, frozenset[str]]:
    """Translate a role into the metadata filter the vector store understands."""
    return {"access": allowed_access_levels(role)}


def retrieve(
    query: str,
    store: VectorStore,
    embedder: Embedder,
    role: str,
    top_k: int,
    min_score: Optional[float] = None,
) -> RetrievalResult:
    """Return up to top_k chunks the given role may read, most similar first."""
    filters = permission_filter(role)          # raises for an unknown role
    query_vector = embedder.embed_query(query)  # 1. embed
    # The filter is never empty here (an unknown role raised above), and the store
    # itself refuses an empty filter, so a bug cannot turn this into an unfiltered search.
    pairs = store.search(                        # 2 + 3 + 4. filter, score, top-k
        query_vector, top_k, filters=filters, min_score=min_score
    )
    hits = [Hit(chunk=chunk, vector_score=score, score=score) for chunk, score in pairs]
    excluded = len(store) - store.count_matching(filters)
    return RetrievalResult(hits=hits, excluded_by_permissions=excluded)
