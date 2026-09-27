"""Embedders: fixed dimension, determinism (also across processes), sensible similarity."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

from rag.embedders import Embedder, HashingEmbedder, get_embedder

HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None


def test_dimension_shape_and_dtype():
    embedder = HashingEmbedder(dimension=256)
    vectors = embedder.embed_documents(["parental leave policy", "expense claim receipts", "x"])
    assert embedder.dimension == 256
    assert vectors.shape == (3, 256)
    assert vectors.dtype == np.float32


def test_rows_are_unit_length():
    vectors = HashingEmbedder().embed_documents(["parental leave policy", "expense claim receipts"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_same_text_gives_identical_vectors():
    embedder = HashingEmbedder()
    first = embedder.embed_query("What is our parental leave policy?")
    second = HashingEmbedder().embed_query("What is our parental leave policy?")
    assert np.array_equal(first, second)


def test_embedding_does_not_depend_on_batch_composition():
    embedder = HashingEmbedder()
    alone = embedder.embed_documents(["parental leave"])[0]
    in_batch = embedder.embed_documents(["something else entirely", "parental leave"])[1]
    assert np.array_equal(alone, in_batch)


def test_vectors_are_identical_in_another_process_with_another_hash_seed(project_root):
    """Python's built-in hash() changes per process; ours must not."""
    script = (
        "import hashlib\n"
        "from rag.embedders import HashingEmbedder\n"
        "v = HashingEmbedder().embed_query('parental leave policy for new parents')\n"
        "print(hashlib.sha256(v.tobytes()).hexdigest())\n"
    )
    digests = set()
    for seed in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=project_root, env=env,
            capture_output=True, text=True, check=True,
        )
        digests.add(result.stdout.strip())
    assert len(digests) == 1


def test_related_text_scores_higher_than_unrelated_text():
    embedder = HashingEmbedder()
    query = embedder.embed_query("How many weeks of parental leave do I get?")
    related = embedder.embed_query("Parental leave: employees get 12 weeks of paid leave.")
    unrelated = embedder.embed_query("Passwords must be at least 14 characters long.")
    assert float(query @ related) > float(query @ unrelated) + 0.1


def test_empty_text_gives_a_zero_vector_not_nan():
    vector = HashingEmbedder().embed_query("")
    assert not np.isnan(vector).any()
    assert float(np.linalg.norm(vector)) == 0.0


def test_stop_words_only_gives_a_zero_vector():
    assert float(np.linalg.norm(HashingEmbedder().embed_query("what is the"))) == 0.0


def test_no_texts_gives_an_empty_matrix():
    assert HashingEmbedder(dimension=64).embed_documents([]).shape == (0, 64)


def test_embed_query_matches_embed_documents():
    embedder = HashingEmbedder()
    assert np.array_equal(embedder.embed_query("expense receipts"),
                          embedder.embed_documents(["expense receipts"])[0])


def test_too_small_dimension_is_rejected():
    with pytest.raises(ValueError):
        HashingEmbedder(dimension=4)


def test_factory_returns_the_hashing_embedder_by_default_name():
    embedder = get_embedder("hashing")
    assert isinstance(embedder, HashingEmbedder)
    assert isinstance(embedder, Embedder)
    assert embedder.name == "hashing"


def test_factory_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown EMBEDDER"):
        get_embedder("word2vec")


@pytest.mark.skipif(HAS_SENTENCE_TRANSFORMERS, reason="only meaningful when sentence-transformers is NOT installed")
def test_optional_embedder_explains_how_to_install_it():
    with pytest.raises(ImportError, match="pip install sentence-transformers"):
        get_embedder("sentence-transformers")
