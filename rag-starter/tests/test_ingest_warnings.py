"""Ingest safety nets: text the default embedder cannot read, and runs that create nothing.

The default hashing embedder only sees ASCII [a-z0-9]+. Non-English text (here:
Nepali, written with \\u escapes so this file stays ASCII) becomes an all-zero
vector, which can never be retrieved. Before this check that happened silently.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rag.config import Config
from rag.embedders import HashingEmbedder
from rag.pipeline import MAX_LISTED_WARNINGS, build_index, build_store, ingest_warnings
from rag.text import tokenize

# "Nepali language" and a short sentence, as \u escapes.
NEPALI_HEADING = "\u0928\u0947\u092a\u093e\u0932\u0940 \u092d\u093e\u0937\u093e"
NEPALI_TEXT = (
    "\u092f\u094b \u0928\u0940\u0924\u093f \u0938\u092c\u0948 "
    "\u0915\u0930\u094d\u092e\u091a\u093e\u0930\u0940\u0939\u0930\u0942\u0915\u094b "
    "\u0932\u093e\u0917\u093f \u0939\u094b\u0964"
)
ENGLISH = "# Office hours\n\nThe office opens at nine in the morning and closes at five.\n"


def write_doc(folder: Path, name: str, body: str, access: str = "all") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(f"---\naccess: {access}\nupdated: 2026-01-01\n---\n{body}".encode("utf-8"))
    return path


def run_ingest(project_root: Path, docs: Path, index: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("EMBEDDER",)}
    return subprocess.run(
        [sys.executable, "ingest.py", "--docs", str(docs), "--index", str(index)],
        cwd=project_root, env=env, capture_output=True, text=True, encoding="utf-8",
    )


# ------------------------------------------------------ the tokenizer limitation


def test_the_tokenizer_returns_nothing_for_non_ascii_scripts():
    assert tokenize(NEPALI_HEADING + " " + NEPALI_TEXT) == []
    assert tokenize("\u4e2d\u6587\u6587\u672c \u0627\u0644\u0639\u0631\u0628\u064a\u0629") == []


def test_non_ascii_text_gives_an_all_zero_vector_with_the_hashing_embedder():
    vector = HashingEmbedder().embed_query(NEPALI_TEXT)
    assert float(np.linalg.norm(vector)) == 0.0


def test_mixed_text_is_indexed_by_its_ascii_words_only():
    assert tokenize(f"Leave 12 {NEPALI_HEADING} weeks") == ["leave", "12", "week"]


# --------------------------------------------------- warnings (in-process)


def test_a_chunk_with_no_ascii_text_is_reported_by_file_and_chunk(tmp_path):
    write_doc(tmp_path / "docs", "office.md", ENGLISH)
    write_doc(tmp_path / "docs", "nepali.md", f"# {NEPALI_HEADING}\n\n{NEPALI_TEXT}\n")
    summary = build_index(tmp_path / "docs", tmp_path / "index", Config())
    assert summary.chunks == 2
    joined = "\n".join(summary.warnings)
    assert "nepali.md, chunk nepali.md::0" in joined
    assert "all zero" in joined
    assert "office.md" not in joined  # the English document is fine
    assert "English/ASCII-oriented" in joined
    assert "sentence-transformers" in joined
    assert "1 of 2 chunk(s)" in joined


def test_warning_lines_stay_ascii_even_for_non_ascii_headings(tmp_path):
    write_doc(tmp_path / "docs", "nepali.md", f"# {NEPALI_HEADING}\n\n{NEPALI_TEXT}\n")
    warnings = build_index(tmp_path / "docs", tmp_path / "index", Config()).warnings
    assert warnings and all(line.isascii() for line in warnings)


def test_non_ascii_text_under_an_english_heading_is_reported_as_heading_only(tmp_path):
    """The heading gives the chunk a vector, but its content cannot be matched."""
    write_doc(tmp_path / "docs", "mixed.md", f"# Leave policy\n\n{NEPALI_TEXT}\n")
    warnings = build_index(tmp_path / "docs", tmp_path / "index", Config()).warnings
    joined = "\n".join(warnings)
    assert "mixed.md, chunk mixed.md::0" in joined
    assert "indexed by its section heading only" in joined


def test_a_normal_english_run_produces_no_warnings(tmp_path, docs_dir):
    summary = build_index(docs_dir, tmp_path / "index", Config())
    assert summary.warnings == ()


def test_many_bad_chunks_are_listed_up_to_a_limit_and_then_summarized():
    from rag.chunking import chunk_documents
    from rag.loaders import Document, Segment

    sections = "\n".join(f"# {NEPALI_HEADING} {i}\n\n{NEPALI_TEXT}\n" for i in range(MAX_LISTED_WARNINGS + 3))
    document = Document("many.md", "many", "all", "", "2026-01-01", (Segment(sections),))
    embedder = HashingEmbedder()
    chunks = chunk_documents([document], 220, 40)
    assert len(chunks) == MAX_LISTED_WARNINGS + 3
    vectors = embedder.embed_documents([chunk.embed_text for chunk in chunks])
    warnings = ingest_warnings([document], chunks, vectors, embedder)
    assert sum(1 for line in warnings if line.startswith("many.md, chunk")) == MAX_LISTED_WARNINGS
    assert any("... and 3 more chunk(s)" in line for line in warnings)
    assert any(f"{MAX_LISTED_WARNINGS + 3} of {MAX_LISTED_WARNINGS + 3} chunk(s)" in line for line in warnings)


def test_a_document_that_creates_no_chunks_is_named_in_a_warning(tmp_path):
    write_doc(tmp_path / "docs", "office.md", ENGLISH)
    write_doc(tmp_path / "docs", "empty.md", "\n")  # front matter only
    summary = build_index(tmp_path / "docs", tmp_path / "index", Config())
    assert summary.documents == 2 and summary.chunks == 1
    assert any(line.startswith("empty.md: no chunks were created") for line in summary.warnings)


def test_a_run_that_creates_no_chunk_at_all_fails(tmp_path):
    write_doc(tmp_path / "docs", "empty.md", "\n")
    with pytest.raises(ValueError, match="No chunks were created from 1 document"):
        build_index(tmp_path / "docs", tmp_path / "index", Config())
    assert not (tmp_path / "index").exists()  # nothing half-built is left behind


def test_build_store_reports_through_the_optional_warnings_list(tmp_path):
    from rag.loaders import load_documents

    write_doc(tmp_path / "docs", "nepali.md", f"# {NEPALI_HEADING}\n\n{NEPALI_TEXT}\n")
    documents = load_documents(tmp_path / "docs")
    collected: list[str] = []
    build_store(documents, HashingEmbedder(), Config(), collected)
    assert collected
    build_store(documents, HashingEmbedder(), Config())  # the list is optional


# -------------------------------------------------------- the ingest.py command


def test_ingest_cli_warns_on_stderr_naming_the_file_and_chunk_and_still_succeeds(tmp_path, project_root):
    write_doc(tmp_path / "docs", "office.md", ENGLISH)
    write_doc(tmp_path / "docs", "nepali.md", f"# {NEPALI_HEADING}\n\n{NEPALI_TEXT}\n")
    result = run_ingest(project_root, tmp_path / "docs", tmp_path / "index")
    assert result.returncode == 0, result.stderr
    assert "warning: nepali.md, chunk nepali.md::0" in result.stderr
    assert "English/ASCII-oriented" in result.stderr
    assert "warning" not in result.stdout  # stdout keeps the normal summary only
    assert result.stdout.isascii() and result.stderr.isascii()
    assert (tmp_path / "index" / "vectors.npz").is_file()


def test_ingest_cli_prints_no_warning_for_the_shipped_documents(tmp_path, project_root):
    result = run_ingest(project_root, project_root / "data" / "docs", tmp_path / "index")
    assert result.returncode == 0, result.stderr
    assert "warning:" not in result.stderr


def test_ingest_cli_fails_with_a_clear_message_when_no_chunk_is_created(tmp_path, project_root):
    write_doc(tmp_path / "docs", "empty.md", "\n")
    result = run_ingest(project_root, tmp_path / "docs", tmp_path / "index")
    assert result.returncode == 1
    assert "error: No chunks were created" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "index").exists()


def test_ingest_cli_rejects_a_duplicated_front_matter_key_and_names_file_and_key(tmp_path, project_root):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "board.md").write_text(
        "---\naccess: executive\ntitle: Board pack\naccess: all\n---\n# Board\n\nSecret text.\n",
        encoding="utf-8",
    )
    result = run_ingest(project_root, docs, tmp_path / "index")
    assert result.returncode == 1
    assert "error:" in result.stderr and "board.md" in result.stderr and "'access'" in result.stderr
    assert "more than once" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "index").exists()  # nothing was indexed, so nothing was published
