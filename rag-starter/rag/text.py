"""Tiny text helpers shared by the chunker, the embedder and the re-ranker.

Keeping them in one place guarantees that "what counts as a word" is the same
everywhere. If the embedder and the re-ranker tokenized differently, their
scores would disagree for reasons that are very hard to debug.

LIMITATION: this tokenizer is English/ASCII-oriented. It only sees the ASCII
characters [a-z0-9]+, so text in other scripts (Nepali, Chinese, Arabic, ...)
produces NO tokens, and an accented letter cuts a word in pieces (an accented
"cafe" becomes the token "caf"). With the default hashing embedder a chunk that
has no ASCII letters or digits gets an all-zero vector and can never be
retrieved. ingest.py warns about those chunks by name; for other languages use a
multilingual model through the sentence-transformers hook
(EMBEDDER=sentence-transformers, see the README).
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Very common words carry almost no meaning for lexical matching. Dropping
# them stops "what is the" from dominating a short question.
STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after all also am an and any are as at be been before but by
    can could did do does for from had has have how i if in into is it its me
    more most my no not of on or our out over should so some than that the
    their them then there these they this to up us was we were what when where
    which who whom why will with would you your
    """.split()
)


def count_words(text: str) -> int:
    """Number of whitespace-separated words (our stand-in for tokens)."""
    return len(text.split())


def fold_plural(token: str) -> str:
    """Very crude plural folding: 'policies' -> 'policy', 'weeks' -> 'week'.

    This is NOT a real stemmer. It only removes the most common plural
    endings so that a question about "expenses" matches text about
    "expense". Real embedding models handle this (and much more) by learning
    from data; that is one reason to plug one in later.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, split into alphanumeric tokens, drop stop words, fold plurals.

    Single letters are dropped (they are mostly the "s" of "what's"), but
    single digits are kept because numbers matter in policy text.

    Only ASCII letters and digits are recognized: text in other scripts returns
    an empty list (see the LIMITATION note at the top of this module).
    """
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if raw in STOPWORDS:
            continue
        if len(raw) == 1 and not raw.isdigit():
            continue
        tokens.append(fold_plural(raw))
    return tokens
