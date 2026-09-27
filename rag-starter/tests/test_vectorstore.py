"""Vector store: search order, metadata filtering before scoring, save/load round trip."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fakes import make_chunk
from rag.vectorstore import CHUNKS_FILE, VECTORS_FILE, IndexMismatchError, VectorStore


def small_store() -> VectorStore:
    """Five chunks; the two BEST matches for query [1, 0, 0] are restricted."""
    chunks = [
        make_chunk(0, section="best restricted", access="executive"),
        make_chunk(1, section="second restricted", access="executive"),
        make_chunk(2, section="good public", access="all"),
        make_chunk(3, section="ok public", access="all"),
        make_chunk(4, section="manager only", access="managers"),
    ]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.7, 0.3, 0.0],
            [0.3, 0.7, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    return VectorStore(chunks, embeddings, embedder_name="test")


QUERY = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def search_everything(store: VectorStore, query, top_k: int, **kwargs):
    """Tests that are not about permissions search every chunk ON PURPOSE."""
    return store.search(query, top_k, filters=None, allow_unfiltered=True, **kwargs)


# ------------------------------------------------------------- fail closed


@pytest.mark.parametrize("filters", [None, {}])
def test_search_without_filters_raises_unless_explicitly_allowed(filters):
    with pytest.raises(ValueError, match="allow_unfiltered"):
        small_store().search(QUERY, top_k=3, filters=filters)


def test_forgetting_the_filters_argument_is_an_error_too():
    with pytest.raises(TypeError):
        small_store().search(QUERY, top_k=3)  # type: ignore[call-arg]


def test_filters_must_be_passed_by_keyword():
    with pytest.raises(TypeError):
        small_store().search(QUERY, 3, {"access": {"all"}})  # type: ignore[misc]


@pytest.mark.parametrize("filters", [None, {}])
def test_allow_unfiltered_searches_every_chunk(filters):
    results = small_store().search(QUERY, top_k=10, filters=filters, allow_unfiltered=True)
    assert len(results) == 5
    assert results[0][0].access == "executive"  # the restricted chunk is the best match


def test_allow_unfiltered_does_not_switch_off_filters_that_are_given():
    results = small_store().search(
        QUERY, top_k=10, filters={"access": {"all"}}, allow_unfiltered=True
    )
    assert {chunk.access for chunk, _ in results} == {"all"}


def test_a_filter_with_an_empty_allowed_set_is_still_a_filter_and_matches_nothing():
    assert small_store().search(QUERY, top_k=5, filters={"access": set()}) == []


def test_the_permission_path_always_passes_a_non_empty_filter():
    """retrieve() must work without allow_unfiltered, for every known role."""
    from rag.config import ROLES
    from rag.embedders import HashingEmbedder
    from rag.retrieve import permission_filter, retrieve

    chunks = [make_chunk(0, text="parental leave", access="all"),
              make_chunk(1, text="executive pay", access="executive")]
    embedder = HashingEmbedder()
    store = VectorStore(chunks, embedder.embed_documents([c.text for c in chunks]), embedder.name)
    for role in ROLES:
        assert permission_filter(role)  # never empty
        retrieve("parental leave", store, embedder, role, top_k=5)


# ------------------------------------------------------------------- search


def test_results_are_sorted_best_first_and_limited_to_top_k():
    results = search_everything(small_store(), QUERY, 3)
    assert [chunk.section for chunk, _ in results] == [
        "best restricted", "second restricted", "good public",
    ]
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_scores_are_cosine_similarities():
    results = dict(
        (chunk.section, score)
        for chunk, score in search_everything(small_store(), QUERY, 5)
    )
    assert results["best restricted"] == pytest.approx(1.0)
    assert results["good public"] == pytest.approx(0.7 / np.sqrt(0.7**2 + 0.3**2))


def test_filter_is_applied_before_top_k():
    """Pre-filtering: asking for 2 results yields 2 ALLOWED results.

    A post-filter (search top 2 first, then drop restricted ones) would return
    nothing here, because the two best matches are both restricted.
    """
    results = small_store().search(QUERY, top_k=2, filters={"access": {"all"}})
    assert [chunk.section for chunk, _ in results] == ["good public", "ok public"]


def test_filtered_out_chunks_never_appear_even_if_they_are_the_best_match():
    results = small_store().search(QUERY, top_k=10, filters={"access": {"all", "managers"}})
    assert {chunk.access for chunk, _ in results} <= {"all", "managers"}
    assert len(results) == 3


def test_multiple_filters_must_all_pass():
    store = small_store()
    results = store.search(
        QUERY, top_k=5, filters={"access": {"all", "executive"}, "section": {"ok public"}}
    )
    assert [chunk.section for chunk, _ in results] == ["ok public"]


def test_unknown_filter_field_raises_instead_of_being_ignored():
    with pytest.raises(ValueError, match="unknown chunk field"):
        small_store().search(QUERY, top_k=3, filters={"acces": {"all"}})


def test_count_matching_reports_how_many_chunks_pass():
    store = small_store()
    assert store.count_matching({"access": {"all"}}) == 2
    assert store.count_matching(None) == 5


def test_min_score_drops_weak_matches():
    results = search_everything(small_store(), QUERY, 5, min_score=0.95)
    assert [chunk.section for chunk, _ in results] == ["best restricted", "second restricted"]


def test_equal_scores_keep_insertion_order():
    chunks = [make_chunk(i, section=f"s{i}") for i in range(3)]
    embeddings = np.array([[1.0, 0.0]] * 3, dtype=np.float32)
    results = search_everything(VectorStore(chunks, embeddings, "test"), np.array([1.0, 0.0]), 3)
    assert [chunk.section for chunk, _ in results] == ["s0", "s1", "s2"]


def test_wrong_query_dimension_raises():
    with pytest.raises(ValueError, match="shape"):
        small_store().search(np.array([1.0, 0.0]), top_k=3, filters={"access": {"all"}})


def test_mismatched_chunks_and_vectors_raise():
    with pytest.raises(ValueError):
        VectorStore([make_chunk(0)], np.zeros((2, 3), dtype=np.float32), "test")


def test_save_and_load_round_trip(tmp_path):
    store = small_store()
    store.build_info = {"chunk_words": 220}
    store.save(tmp_path / "index")

    assert (tmp_path / "index" / VECTORS_FILE).is_file()
    assert (tmp_path / "index" / CHUNKS_FILE).is_file()

    loaded = VectorStore.load(tmp_path / "index")
    assert loaded.chunks == store.chunks
    assert loaded.embedder_name == "test"
    assert loaded.dimension == 3
    assert loaded.build_info == {"chunk_words": 220}
    assert np.allclose(loaded._embeddings, store._embeddings)

    before = [(c.chunk_id, round(s, 6)) for c, s in search_everything(store, QUERY, 5)]
    after = [(c.chunk_id, round(s, 6)) for c, s in search_everything(loaded, QUERY, 5)]
    assert before == after


def test_saved_chunks_json_is_readable_and_keeps_metadata(tmp_path):
    small_store().save(tmp_path)
    payload = json.loads((tmp_path / CHUNKS_FILE).read_text(encoding="utf-8"))
    assert payload["embedder"] == "test"
    assert payload["chunks"][0]["access"] == "executive"
    assert payload["chunks"][0]["source"] == "doc.md"


def test_save_creates_missing_parent_folders(tmp_path):
    small_store().save(tmp_path / "a" / "b" / "index")
    assert (tmp_path / "a" / "b" / "index" / VECTORS_FILE).is_file()


def test_loading_a_missing_index_explains_how_to_build_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest.py") as excinfo:
        VectorStore.load(tmp_path / "nothing-here")
    assert "Run this from inside the rag-starter folder." in str(excinfo.value)


def test_a_chunks_file_that_is_not_json_gives_a_friendly_error(tmp_path):
    small_store().save(tmp_path)
    (tmp_path / CHUNKS_FILE).write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON.*ingest.py"):
        VectorStore.load(tmp_path)


def test_a_chunks_file_that_is_not_an_object_gives_a_friendly_error(tmp_path):
    small_store().save(tmp_path)
    (tmp_path / CHUNKS_FILE).write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="does not look like an index"):
        VectorStore.load(tmp_path)


def test_embedder_mismatch_is_detected():
    store = small_store()
    store.check_embedder("test", 3)  # fine
    with pytest.raises(IndexMismatchError, match="ingest.py"):
        store.check_embedder("other-model", 3)
    with pytest.raises(IndexMismatchError):
        store.check_embedder("test", 8)


def test_empty_store_returns_no_results():
    store = VectorStore([], np.zeros((0, 3), dtype=np.float32), "test")
    assert store.search(QUERY, top_k=5, filters={"access": {"all"}}) == []
