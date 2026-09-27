"""Settings, roles and the permission model.

Two ideas live here:

1. Tunable settings (chunk size, top-k, ...) read from environment variables, so
   the same code runs on a laptop, in CI and in a container without edits.
2. The role -> access-level mapping. This tiny table IS the access-control
   policy of the demo. In a real system it would come from your identity
   provider (groups, claims, ...) - but the mechanism stays the same: the
   caller's permissions are turned into a metadata filter that is applied
   BEFORE any chunk is scored or shown to the LLM.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

# Every document declares one of these in its front matter ("access: all").
ACCESS_LEVELS: tuple[str, ...] = ("all", "managers", "executive")

# A role may read every document whose access level is listed for that role.
ROLE_ACCESS: dict[str, frozenset[str]] = {
    "employee": frozenset({"all"}),
    "manager": frozenset({"all", "managers"}),
    "executive": frozenset({"all", "managers", "executive"}),
}

ROLES: tuple[str, ...] = tuple(ROLE_ACCESS)


def allowed_access_levels(role: str) -> frozenset[str]:
    """Return the access levels a role may read.

    An unknown role raises instead of falling back to some default. Failing
    closed is the safe behavior for anything security related: a typo in a
    role name must never silently widen (or bypass) the filter.
    """
    try:
        return ROLE_ACCESS[role]
    except KeyError:
        known = ", ".join(ROLES)
        raise ValueError(f"Unknown role {role!r}. Known roles: {known}.") from None


@dataclass(frozen=True)
class Config:
    """All tunable settings, with defaults that suit the small demo corpus.

    Sizes are counted in WORDS because words need no tokenizer. As a rough
    rule of thumb one token is about 0.75 English words, so 220 words is
    roughly 300 tokens. Measure with your own model's tokenizer before you
    rely on that ratio.
    """

    embedder: str = "hashing"      # "hashing" (offline) or "sentence-transformers"
    sentence_model: str = "all-MiniLM-L6-v2"  # only used by sentence-transformers
    chunk_words: int = 220         # target maximum chunk size, in words
    chunk_overlap: int = 40        # words repeated between neighboring chunks
    top_k: int = 20                # candidates fetched from the vector store
    rerank_top_n: int = 5          # chunks kept after re-ranking (go to the LLM)
    min_score: float = 0.05        # candidates below this similarity are dropped
    llm_model: str = "claude-opus-5"

    def __post_init__(self) -> None:
        if self.chunk_words < 1:
            raise ValueError("CHUNK_WORDS must be at least 1.")
        if not 0 <= self.chunk_overlap < self.chunk_words:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_WORDS.")
        if self.top_k < 1:
            raise ValueError("TOP_K must be at least 1.")
        if self.rerank_top_n < 1:
            raise ValueError("RERANK_TOP_N must be at least 1.")
        # float() happily accepts "nan" and "inf": nan would make the score cut-off
        # silently do nothing, inf would silently drop every result.
        if not math.isfinite(self.min_score):
            raise ValueError(f"MIN_SCORE must be a finite number, got {self.min_score!r}.")

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "Config":
        """Build a Config from environment variables (see .env.example).

        Empty values count as "not set", so copying .env.example to .env and
        leaving a line blank keeps the default.
        """
        env = os.environ if environ is None else environ
        defaults = cls()

        def text(name: str, default: str) -> str:
            value = env.get(name, "").strip()
            return value or default

        def number(name: str, default, convert):
            raw = env.get(name, "").strip()
            if not raw:
                return default
            try:
                return convert(raw)
            except ValueError:
                raise ValueError(f"{name} must be a number, got {raw!r}.") from None

        return cls(
            embedder=text("EMBEDDER", defaults.embedder).lower(),
            sentence_model=text("SENTENCE_MODEL", defaults.sentence_model),
            chunk_words=number("CHUNK_WORDS", defaults.chunk_words, int),
            chunk_overlap=number("CHUNK_OVERLAP", defaults.chunk_overlap, int),
            top_k=number("TOP_K", defaults.top_k, int),
            rerank_top_n=number("RERANK_TOP_N", defaults.rerank_top_n, int),
            min_score=number("MIN_SCORE", defaults.min_score, float),
            llm_model=text("LLM_MODEL", defaults.llm_model),
        )


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines from a .env file into os.environ.

    A ten-line replacement for the python-dotenv package, so the template has
    one dependency less. Variables that are already set (and non-empty) win,
    which lets you override any value from the shell for a single run.
    A missing file is fine: the defaults still work.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and not os.environ.get(key):
            os.environ[key] = value
