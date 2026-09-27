"""Step 3 of the pipeline: turn text into vectors ("embeddings").

An embedder maps a piece of text to a list of numbers so that texts about the
same thing end up close together. Retrieval then becomes "find the stored
vectors closest to the question's vector".

Two implementations
    HashingEmbedder            DEFAULT. Offline, deterministic, no model to
                               download. It is LEXICAL: it matches shared
                               words, not meaning. Perfect for a demo and for
                               tests; a real system should use a learned model.
    SentenceTransformerEmbedder  OPTIONAL. A real semantic model. It needs
                               `pip install sentence-transformers` and, on first
                               use, downloads the model from the Hugging Face hub.

Every embedder returns float32 vectors that are L2-normalized (length 1), so
"cosine similarity" is simply a dot product.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import Sequence

import numpy as np

from .text import tokenize


class Embedder(ABC):
    """The interface every embedder implements.

    To plug in a different model (OpenAI, Cohere, a local ONNX model, ...),
    subclass this, set `name` and `dimension`, and implement embed_documents.
    """

    name: str
    dimension: int

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dimension), float32, unit-length rows."""

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single question.

        Separate from embed_documents because some models want a different
        prefix or instruction for queries than for documents. Override this if
        yours does.
        """
        return self.embed_documents([text])[0]


def _stable_hash(feature: str) -> int:
    """A 64-bit hash that is the same on every machine and in every run.

    Python's built-in hash() is deliberately randomized per process (for
    security), so an index built today would not match queries embedded
    tomorrow. blake2b from hashlib is stable.
    """
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class HashingEmbedder(Embedder):
    """Bag-of-words "feature hashing" embedder (the hashing trick).

    How it works
      1. Tokenize the text (lowercase, drop stop words, fold plurals).
      2. Make features: every word, plus every pair of neighboring words
         (bigrams like "parental_leave") at half weight, which rewards phrase
         matches.
      3. Hash each feature to one of `dimension` slots and add +/- its weight
         there. The random sign makes hash collisions cancel out on average
         instead of piling up.
      4. Normalize the vector to length 1.

    It has a fixed dimension whatever the vocabulary, needs no training and
    no download, and gives identical results everywhere. What it cannot do is
    understand synonyms: "doctor's note" does not match "medical certificate".

    Two unrelated words sometimes land in the same slot ("collide") and then
    look slightly similar. A larger `dimension` makes that rarer at the cost of
    memory: on the four demo documents a 1024-slot vector produced noticeably
    more accidental similarity than the 4096-slot default. Even so, small
    similarity scores from this embedder are partly noise; real embedding
    models separate related from unrelated text much more cleanly.
    """

    name = "hashing"

    def __init__(self, dimension: int = 4096, bigram_weight: float = 0.5) -> None:
        if dimension < 16:
            raise ValueError("dimension must be at least 16")
        self.dimension = dimension
        self.bigram_weight = bigram_weight

    def _embed_one(self, text: str) -> np.ndarray:
        tokens = tokenize(text)
        unigrams = Counter(tokens)
        bigrams = Counter(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))

        vector = np.zeros(self.dimension, dtype=np.float32)
        for counter, base_weight in ((unigrams, 1.0), (bigrams, self.bigram_weight)):
            for feature, count in counter.items():
                hashed = _stable_hash(feature)
                slot = hashed % self.dimension
                sign = 1.0 if (hashed >> 63) & 1 else -1.0
                # 1 + log(count): repeating a word helps, but with diminishing returns.
                vector[slot] += sign * base_weight * (1.0 + math.log(count))

        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector  # all-zero for empty text: it matches nothing, by design

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._embed_one(text) for text in texts]).astype(np.float32)


class SentenceTransformerEmbedder(Embedder):
    """Semantic embeddings from the `sentence-transformers` library (optional).

    Install with `pip install sentence-transformers`. The first use downloads
    the model (all-MiniLM-L6-v2 is roughly 90 MB) into the Hugging Face cache;
    set the HF_HOME environment variable to choose where that cache lives.
    The library is imported inside __init__, so the rest of the project works
    without it installed.

    An index is tied to the embedder that built it: if you change the model you
    must run ingest.py again (the index stores the embedder name and refuses to
    be searched with a different one).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "EMBEDDER=sentence-transformers needs the optional package: "
                "pip install sentence-transformers"
            ) from exc
        self.name = f"sentence-transformers:{model_name}"
        self._model = SentenceTransformer(model_name)
        # Ask the model for one embedding instead of a version-specific method.
        self.dimension = int(self.embed_documents(["dimension probe"]).shape[1])

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype=np.float32)


def get_embedder(name: str, sentence_model: str = "all-MiniLM-L6-v2") -> Embedder:
    """Create the embedder selected by the EMBEDDER setting."""
    key = name.strip().lower()
    if key == "hashing":
        return HashingEmbedder()
    if key in ("sentence-transformers", "sentence_transformers", "st"):
        return SentenceTransformerEmbedder(sentence_model)
    raise ValueError(
        f"Unknown EMBEDDER {name!r}. Use 'hashing' or 'sentence-transformers'."
    )
