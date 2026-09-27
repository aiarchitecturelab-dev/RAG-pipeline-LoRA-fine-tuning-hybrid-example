"""Step 2 of the pipeline: cut documents into chunks.

Why chunk at all? A retriever compares the question with a piece of text, and
the LLM has a limited, paid-for context window. A whole 40-page policy is too
big and too vague to be a good match for one question; a paragraph about
parental leave is a precise one. Chunk size is a real trade-off:

    too small  -> the answer gets cut in half, and each chunk lacks context
    too large  -> the chunk is "about many things", its score is diluted, and
                  you pay to send text the model does not need

The video's rule of thumb is roughly 300-800 tokens. We count WORDS instead
(no tokenizer needed; about 0.75 words per token in English). Run
evals/eval_retrieval.py with different --chunk-words values to see the effect
on your own documents rather than trusting any rule of thumb.

What this chunker does
    * splits a markdown document at its headings first, so a chunk never mixes
      two topics, and remembers the section heading in each chunk's metadata;
    * inside a section it packs whole sentences (or whole list items / table
      rows) up to the size limit, so a chunk never ends mid-sentence unless a
      single sentence is longer than the limit;
    * repeats a little text ("overlap") from the end of one chunk at the start
      of the next, so a fact that sits on a boundary is still found whole.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Optional

from .loaders import Document
from .text import count_words

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_LIST_ITEM_RE = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+|\|)")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

# A full stop after these does not end a sentence ("e.g. this", "approx. 5").
_ABBREVIATIONS = ("e.g.", "i.e.", "vs.", "approx.", "no.", "mr.", "mrs.", "ms.", "dr.")


@dataclass(frozen=True)
class Chunk:
    """One retrievable piece of text plus everything we know about it.

    The metadata (source, section, page, access, ...) travels with the text
    into the vector store. It is what lets us filter by permission and build
    citations later.
    """

    chunk_id: str
    source: str               # file name, e.g. "hr-policy-2026.md"
    section: str              # nearest heading ("" for PDF pages)
    page: Optional[int]       # PDF page number, else None
    text: str
    access: str
    department: str
    updated: str

    @property
    def locator(self) -> str:
        """Where inside the source this chunk lives: "p.12" or the section name."""
        if self.page is not None:
            return f"p.{self.page}"
        return self.section

    @property
    def embed_text(self) -> str:
        """The text that gets embedded: heading + body.

        A chunk cut out of its section loses the heading that told the reader
        what it is about. Putting the heading back in front of the text before
        embedding is a cheap and effective way to keep that context.
        """
        return f"{self.section}\n{self.text}" if self.section else self.text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**data)


# --------------------------------------------------------------------------
# Splitting a document into sections, and a section into sentence-like units
# --------------------------------------------------------------------------

def split_sections(text: str, default_heading: str = "") -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs at every '#' heading.

    Text before the first heading is filed under `default_heading`. Only the
    nearest heading is kept (not the full "Parent > Child" path) to keep
    citations short. Sections with an empty body are dropped.
    """
    sections: list[tuple[str, list[str]]] = [(default_heading, [])]
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            sections.append((match.group(2).strip(), []))
        else:
            sections[-1][1].append(line)
    result = []
    for heading, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if body:
            result.append((heading, body))
    return result


def is_list_line(line: str) -> bool:
    """True for bullet items, numbered items and table rows."""
    return bool(_LIST_ITEM_RE.match(line.strip()))


def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences with a simple punctuation rule.

    Good enough for policy prose. It is not a linguistic parser: unusual
    abbreviations can still cause a wrong split, which merely produces a
    slightly early chunk boundary - not a lost sentence.
    """
    pieces = _SENTENCE_END_RE.split(paragraph.strip())
    sentences: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        # Compare the LAST WORD only, so "casino." is not mistaken for "no.".
        if sentences and sentences[-1].split()[-1].lower() in _ABBREVIATIONS:
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


def split_units(body: str) -> list[str]:
    """Break a section body into "units": sentences, list items, table rows.

    Units are the smallest pieces the chunker will not cut apart.
    """
    units: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            units.extend(split_sentences(" ".join(paragraph)))
            paragraph.clear()

    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
        elif is_list_line(stripped):
            flush_paragraph()
            units.append(stripped)
        else:
            paragraph.append(stripped)
    flush_paragraph()
    return units


def join_units(units: list[str]) -> str:
    """Re-join units: a space between sentences, a newline around list/table lines."""
    text = ""
    for unit in units:
        if not text:
            text = unit
        elif is_list_line(unit) or is_list_line(text.rsplit("\n", 1)[-1]):
            text = f"{text}\n{unit}"
        else:
            text = f"{text} {unit}"
    return text


# --------------------------------------------------------------------------
# Packing units into size-limited, overlapping chunks
# --------------------------------------------------------------------------

def _split_long_unit(unit: str, chunk_words: int, overlap_words: int) -> list[str]:
    """Last resort for a single unit longer than chunk_words: cut it by words."""
    words = unit.split()
    step = max(1, chunk_words - overlap_words)
    pieces = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start:start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return pieces


def pack_units(units: list[str], chunk_words: int, overlap_words: int) -> list[str]:
    """Greedily pack units into chunks of at most `chunk_words` words.

    Guarantees:
      * every chunk has between 1 and `chunk_words` words (never empty);
      * chunks only end at a unit boundary, except for a unit that alone is
        longer than `chunk_words`;
      * the start of each chunk repeats the last whole units of the previous
        chunk, up to `overlap_words` words (never the entire previous chunk).
    """
    if chunk_words < 1:
        raise ValueError("chunk_words must be at least 1")
    if not 0 <= overlap_words < chunk_words:
        raise ValueError("overlap_words must be >= 0 and smaller than chunk_words")

    # First make sure no single unit is bigger than a chunk.
    sized: list[str] = []
    for unit in units:
        if count_words(unit) > chunk_words:
            sized.extend(_split_long_unit(unit, chunk_words, overlap_words))
        else:
            sized.append(unit)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for unit in sized:
        unit_words = count_words(unit)
        if current and current_words + unit_words > chunk_words:
            chunks.append(join_units(current))
            current, current_words = _overlap_tail(current, overlap_words, unit_words, chunk_words)
        current.append(unit)
        current_words += unit_words
    if current:
        chunks.append(join_units(current))
    return [chunk for chunk in chunks if chunk.strip()]


def _overlap_tail(
    previous_units: list[str], overlap_words: int, next_unit_words: int, chunk_words: int
) -> tuple[list[str], int]:
    """Pick the trailing units of the finished chunk to repeat in the next one."""
    tail: list[str] = []
    tail_words = 0
    # Walk backwards, but never take the whole previous chunk as overlap.
    for unit in reversed(previous_units[1:]):
        unit_words = count_words(unit)
        if tail_words + unit_words > overlap_words:
            break
        tail.insert(0, unit)
        tail_words += unit_words
    # If the overlap plus the next unit would not fit, drop the overlap.
    if tail_words + next_unit_words > chunk_words:
        return [], 0
    return tail, tail_words


def chunk_document(document: Document, chunk_words: int, overlap_words: int) -> list[Chunk]:
    """Turn one Document into Chunks that carry the document's metadata."""
    chunks: list[Chunk] = []
    for segment in document.segments:
        if segment.page is not None:
            sections = [("", segment.text.strip())] if segment.text.strip() else []
        else:
            sections = split_sections(segment.text, default_heading=document.title)
        for heading, body in sections:
            for text in pack_units(split_units(body), chunk_words, overlap_words):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.source}::{len(chunks)}",
                        source=document.source,
                        section=heading,
                        page=segment.page,
                        text=text,
                        access=document.access,
                        department=document.department,
                        updated=document.updated,
                    )
                )
    return chunks


def chunk_documents(
    documents: list[Document], chunk_words: int, overlap_words: int
) -> list[Chunk]:
    """Chunk every document; chunk ids stay unique because they include the source."""
    all_chunks: list[Chunk] = []
    for document in documents:
        all_chunks.extend(chunk_document(document, chunk_words, overlap_words))
    return all_chunks
