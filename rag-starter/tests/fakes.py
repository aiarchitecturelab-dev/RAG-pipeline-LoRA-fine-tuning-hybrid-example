"""Small fake objects used by several test modules.

FakeClient imitates the only part of the Anthropic SDK that rag/llm.py uses:
client.messages.create(...). It records the keyword arguments it was called
with, so tests can check the exact request shape, and it either returns a
canned response or raises a canned exception.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from rag.chunking import Chunk
from rag.retrieve import Hit


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def thinking_block(thinking: str = "let me think") -> SimpleNamespace:
    # Real thinking blocks have no .text attribute at all.
    return SimpleNamespace(type="thinking", thinking=thinking, signature="sig")


def fake_response(blocks: list, stop_reason: str = "end_turn", **extra: Any) -> SimpleNamespace:
    return SimpleNamespace(content=blocks, stop_reason=stop_reason, **extra)


class FakeClient:
    """Stands in for anthropic.Anthropic()."""

    def __init__(self, response: Optional[Any] = None, error: Optional[BaseException] = None):
        self._response = response
        self._error = error
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


def make_chunk(
    index: int = 0,
    source: str = "doc.md",
    section: str = "Section",
    text: str = "some text",
    access: str = "all",
    page: Optional[int] = None,
) -> Chunk:
    return Chunk(
        chunk_id=f"{source}::{index}",
        source=source,
        section=section,
        page=page,
        text=text,
        access=access,
        department="",
        updated="2026-01-01",
    )


def make_hit(chunk: Chunk, score: float = 0.5) -> Hit:
    return Hit(chunk=chunk, vector_score=score, score=score)
