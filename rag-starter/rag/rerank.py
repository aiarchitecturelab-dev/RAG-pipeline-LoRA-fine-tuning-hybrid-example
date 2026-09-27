"""Step 6 of the pipeline: re-rank the candidates.

Why re-rank? The vector search is fast but coarse: it compares one vector per
chunk. A re-ranker looks at the question and each candidate together, which is
slower but usually more accurate. The classic recipe from the video:

    retrieve the top ~50 cheaply  ->  re-score them  ->  keep the best ~5

Only those best few go into the prompt, which keeps the prompt small and the
answer focused.

Rerankers provided:

    LexicalReranker        DEFAULT. Pure Python, offline. Scores how many of
                           the question's words the chunk (and its heading)
                           contains. Simple and explainable, not state of the art.
    PassthroughReranker    Does nothing (keeps the vector order). A baseline.
    CrossEncoderReranker   OPTIONAL HOOK. Wraps a sentence-transformers
                           cross-encoder model. Not covered by this repository's
                           test-suite because it needs a model download.

To use your own, subclass Reranker and implement `rerank`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Sequence

from .retrieve import Hit
from .text import tokenize


class Reranker(ABC):
    """Interface: re-score hits for a question and return them best-first."""

    @abstractmethod
    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        """Return ALL the hits with updated `score`, sorted best first.

        Trimming to the final top-n is done by the caller (see pipeline.py).
        """


class PassthroughReranker(Reranker):
    """Keep the vector-search order (i.e. "no re-ranking").

    Useful as a baseline: evals/eval_retrieval.py --no-rerank uses it to show what
    re-ranking adds (or costs) on your data. With a strong semantic embedder the
    simple lexical re-ranker below can even make the order worse, so measure.
    """

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        return list(hits)


class LexicalReranker(Reranker):
    """Blend of vector score, question-word coverage and heading match.

    score = 0.4 * vector_score + 0.4 * coverage + 0.2 * heading_coverage

    coverage          share of the question's (stop-word-free) words that
                      appear anywhere in the chunk text or heading
    heading_coverage  share of them that appear in the section heading

    The weights are illustrative defaults, not tuned values. Change them and
    re-run evals/eval_retrieval.py to see what happens on your own data.
    """

    def __init__(
        self,
        vector_weight: float = 0.4,
        coverage_weight: float = 0.4,
        heading_weight: float = 0.2,
    ) -> None:
        self.vector_weight = vector_weight
        self.coverage_weight = coverage_weight
        self.heading_weight = heading_weight

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        query_terms = set(tokenize(query))
        rescored: list[Hit] = []
        for hit in hits:
            if query_terms:
                body_terms = set(tokenize(hit.chunk.embed_text))
                heading_terms = set(tokenize(hit.chunk.section))
                coverage = len(query_terms & body_terms) / len(query_terms)
                heading_coverage = len(query_terms & heading_terms) / len(query_terms)
            else:
                coverage = heading_coverage = 0.0
            score = (
                self.vector_weight * hit.vector_score
                + self.coverage_weight * coverage
                + self.heading_weight * heading_coverage
            )
            rescored.append(replace(hit, score=score))
        # sorted() is stable: equal scores keep the retrieval order.
        return sorted(rescored, key=lambda hit: hit.score, reverse=True)


class CrossEncoderReranker(Reranker):
    """Re-rank with a cross-encoder model (optional; needs sentence-transformers).

    A cross-encoder reads the (question, chunk) pair together and outputs one
    relevance score. "cross-encoder/ms-marco-MiniLM-L-6-v2" is a small,
    commonly used choice (roughly 90 MB, downloaded on first use). To enable it,
    build the pipeline with:

        RagPipeline.from_index("index", config, reranker=CrossEncoderReranker())

    Scores from a cross-encoder are on a different scale than cosine
    similarities, so they are not comparable with `vector_score` (and this
    class is not covered by this repository's tests: it needs a model download).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderReranker needs the optional package: "
                "pip install sentence-transformers"
            ) from exc
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        if not hits:
            return []
        pairs = [(query, hit.chunk.embed_text) for hit in hits]
        scores = self._model.predict(pairs)
        rescored = [replace(hit, score=float(score)) for hit, score in zip(hits, scores)]
        return sorted(rescored, key=lambda hit: hit.score, reverse=True)
