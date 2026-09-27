"""Step 1 of the pipeline: read documents and their metadata.

Every document must declare WHO may read it (its "access" level). We attach
that label to the document here, and the chunker copies it onto every chunk.
That is what makes permission filtering at retrieval time possible later:
a chunk that does not carry its access level cannot be filtered.

Supported inputs
    *.md, *.txt   text with an optional "front matter" block at the top:

                      ---
                      title: Acme Corp HR Policy 2026
                      access: all
                      department: Human Resources
                      updated: 2026-01-15
                      ---

    *.pdf         optional, needs the `pypdf` package. PDFs have no front
                  matter, so put the same "key: value" lines in a sidecar file
                  next to it, named like the PDF plus ".meta"
                  (for example HR-Policy-2026.pdf.meta). Page numbers are kept
                  so that citations can say "p.12".

Fail closed: a document without a valid access level is REJECTED. We never
guess a default, because a wrong guess would silently expose a document. For
the same reason a document that repeats a front matter key is REJECTED too: a
second "access:" line must never quietly override the first.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import ACCESS_LEVELS

TEXT_SUFFIXES = (".md", ".txt")
PDF_SUFFIX = ".pdf"


@dataclass(frozen=True)
class Segment:
    """A piece of a document that has one page number (or none).

    Markdown and text files are a single segment with page=None. A PDF has
    one segment per page so the page number survives chunking.
    """

    text: str
    page: Optional[int] = None


@dataclass(frozen=True)
class Document:
    """A loaded document: its metadata plus the text to be chunked."""

    source: str              # file name used in citations, e.g. "hr-policy-2026.md"
    title: str
    access: str              # one of config.ACCESS_LEVELS
    department: str
    updated: str             # ISO date, e.g. "2026-01-15"
    segments: tuple[Segment, ...] = field(default_factory=tuple)


def parse_front_matter(
    raw_text: str, source: str = "front matter"
) -> tuple[dict[str, str], str]:
    """Split "---\\nkey: value\\n---\\nbody" into ({"key": "value"}, "body").

    Deliberately tiny: one "key: value" per line, no nesting, no lists. That is
    all the demo needs, and it avoids a YAML dependency. If the text does not
    start with a front matter block, the metadata dict is empty.

    `source` is only used in error messages (the file name). A key that appears
    twice raises ValueError, see _parse_key_values().
    """
    lines = raw_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw_text
    for end in range(1, len(lines)):
        if lines[end].strip() == "---":
            metadata = _parse_key_values(lines[1:end], source)
            body = "\n".join(lines[end + 1:])
            return metadata, body
    return {}, raw_text  # opening "---" without a closing one: treat as plain text


def _parse_key_values(lines: list[str], source: str = "front matter") -> dict[str, str]:
    """Read "key: value" lines. Keys are case-insensitive; a repeated key is an error.

    Fail closed: if "access: executive" is followed by "access: all", silently
    keeping the last one would publish a restricted document to everybody (and
    keeping the first would hide the mistake). Any repeated key, not only
    "access", raises ValueError naming the file and the key, so a human fixes it.
    """
    metadata: dict[str, str] = {}
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if separator:
            name = key.strip().lower()
            if name in metadata:
                raise ValueError(
                    f"{source}: the key {name!r} appears more than once in the front "
                    "matter. Keep exactly one line per key; a repeated 'access' line "
                    "could silently change who may read the document."
                )
            metadata[name] = value.strip()
    return metadata


def _validate_metadata(metadata: dict[str, str], source: str) -> tuple[str, str]:
    """Check access level and date; return (access, updated). Raises ValueError."""
    access = metadata.get("access", "").lower()
    if not access:
        raise ValueError(
            f"{source}: missing 'access' in the front matter. Every document must "
            f"declare who may read it (one of: {', '.join(ACCESS_LEVELS)})."
        )
    if access not in ACCESS_LEVELS:
        raise ValueError(
            f"{source}: unknown access level {access!r}. "
            f"Use one of: {', '.join(ACCESS_LEVELS)}."
        )
    updated = metadata.get("updated", "")
    if updated:
        try:
            datetime.date.fromisoformat(updated)
        except ValueError:
            raise ValueError(
                f"{source}: 'updated' must be an ISO date like 2026-01-15, got {updated!r}."
            ) from None
    return access, updated


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


def load_text_document(path: Path, source: str) -> Document:
    """Load a .md or .txt file with optional front matter."""
    # utf-8-sig silently removes the BOM that Windows Notepad likes to add.
    raw_text = path.read_text(encoding="utf-8-sig")
    metadata, body = parse_front_matter(raw_text, source)
    access, updated = _validate_metadata(metadata, source)
    title = metadata.get("title") or _first_heading(body) or path.stem
    return Document(
        source=source,
        title=title,
        access=access,
        department=metadata.get("department", ""),
        updated=updated,
        segments=(Segment(text=body, page=None),),
    )


def load_pdf_document(path: Path, source: str) -> Document:
    """Load a PDF page by page (needs `pip install pypdf`).

    Text extraction from PDFs is imperfect: scanned pages have no text layer
    and multi-column layouts can come out in the wrong order. Always spot-check
    what pypdf returns for your own documents.
    """
    try:
        from pypdf import PdfReader  # imported lazily: pypdf is an optional extra
    except ImportError as exc:
        raise ImportError(
            f"{source}: reading PDFs needs the optional 'pypdf' package "
            "(pip install pypdf)."
        ) from exc

    sidecar = path.with_name(path.name + ".meta")
    if not sidecar.is_file():
        raise ValueError(
            f"{source}: PDFs need a metadata file named {sidecar.name} next to them, "
            "with lines like 'access: all' and 'updated: 2026-01-15'."
        )
    metadata = _parse_key_values(
        sidecar.read_text(encoding="utf-8-sig").splitlines(), f"{source}.meta"
    )
    access, updated = _validate_metadata(metadata, source)

    reader = PdfReader(str(path))
    segments = []
    for page_number, page in enumerate(reader.pages, start=1):  # PDF pages count from 1
        page_text = (page.extract_text() or "").strip()
        if page_text:
            segments.append(Segment(text=page_text, page=page_number))
    return Document(
        source=source,
        title=metadata.get("title") or path.stem,
        access=access,
        department=metadata.get("department", ""),
        updated=updated,
        segments=tuple(segments),
    )


def load_documents(docs_dir: str | Path) -> list[Document]:
    """Load every supported file under docs_dir (recursively, in sorted order).

    Sorted order makes ingestion deterministic: the same folder always yields
    the same chunk order and the same chunk ids.
    """
    root = Path(docs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Documents folder not found: {root}")

    documents: list[Document] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        source = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            documents.append(load_text_document(path, source))
        elif suffix == PDF_SUFFIX:
            documents.append(load_pdf_document(path, source))
        # anything else (including .meta sidecars) is ignored on purpose
    if not documents:
        raise FileNotFoundError(f"No .md, .txt or .pdf documents found in {root}")
    return documents
