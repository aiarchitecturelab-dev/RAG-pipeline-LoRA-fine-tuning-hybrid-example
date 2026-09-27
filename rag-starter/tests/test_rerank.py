"""The lexical re-ranker and the optional cross-encoder hook."""

from __future__ import annotations

import importlib.util

import pytest

from fakes import make_chunk, make_hit
from rag.rerank import CrossEncoderReranker, LexicalReranker, PassthroughReranker, Reranker

HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None


def test_a_chunk_that_contains_the_question_words_is_promoted():
    vague = make_hit(make_chunk(0, section="Overview", text="General company information."), 0.30)
    exact = make_hit(make_chunk(1, section="Parental leave", text="Twelve weeks of paid leave."), 0.20)
    ranked = LexicalReranker().rerank("How long is parental leave?", [vague, exact])
    assert [hit.chunk.section for hit in ranked] == ["Parental leave", "Overview"]


def test_reranking_returns_all_hits_sorted_and_keeps_the_original_vector_score():
    hits = [make_hit(make_chunk(i, text=f"text {i}"), 0.1 * i) for i in range(1, 5)]
    ranked = LexicalReranker().rerank("text", hits)
    assert len(ranked) == 4
    scores = [hit.score for hit in ranked]
    assert scores == sorted(scores, reverse=True)
    assert {hit.vector_score for hit in ranked} == {hit.vector_score for hit in hits}


def test_the_heading_match_adds_to_the_score():
    with_heading = make_hit(make_chunk(0, section="Expense receipts", text="Keep them."), 0.2)
    without = make_hit(make_chunk(1, section="Other", text="Expense receipts: keep them."), 0.2)
    ranked = LexicalReranker().rerank("expense receipts", [without, with_heading])
    assert ranked[0].chunk.section == "Expense receipts"


def test_a_question_of_only_stop_words_falls_back_to_the_vector_order():
    hits = [make_hit(make_chunk(0, text="alpha"), 0.5), make_hit(make_chunk(1, text="beta"), 0.4)]
    ranked = LexicalReranker().rerank("what is the", hits)
    assert [hit.chunk.chunk_id for hit in ranked] == ["doc.md::0", "doc.md::1"]


def test_passthrough_reranker_keeps_the_vector_order_and_scores():
    hits = [make_hit(make_chunk(0, text="alpha"), 0.5), make_hit(make_chunk(1, text="beta"), 0.4)]
    ranked = PassthroughReranker().rerank("beta beta beta", hits)
    assert ranked == hits


def test_no_hits_gives_no_results():
    assert LexicalReranker().rerank("anything", []) == []


def test_lexical_reranker_is_a_reranker():
    assert isinstance(LexicalReranker(), Reranker)


@pytest.mark.skipif(HAS_SENTENCE_TRANSFORMERS, reason="only meaningful when sentence-transformers is NOT installed")
def test_cross_encoder_hook_explains_how_to_install_its_dependency():
    with pytest.raises(ImportError, match="pip install sentence-transformers"):
        CrossEncoderReranker()
