"""Loaders: front matter parsing, fail-closed access levels, optional PDF support."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from rag.loaders import load_documents, load_text_document, parse_front_matter

HAS_PYPDF = importlib.util.find_spec("pypdf") is not None


def write(path: Path, text: str, newline: str = "\n") -> Path:
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))
    return path


GOOD = "---\ntitle: My Doc\naccess: managers\ndepartment: Finance\nupdated: 2026-03-01\n---\n# My Doc\n\nBody text.\n"


def test_front_matter_is_parsed_and_removed_from_the_body():
    metadata, body = parse_front_matter(GOOD)
    assert metadata == {
        "title": "My Doc", "access": "managers", "department": "Finance", "updated": "2026-03-01",
    }
    assert body.lstrip().startswith("# My Doc")
    assert "access:" not in body


def test_text_without_front_matter_gives_empty_metadata():
    assert parse_front_matter("# Just a heading\nText") == ({}, "# Just a heading\nText")


def test_unclosed_front_matter_is_treated_as_plain_text():
    text = "---\naccess: all\nno closing marker"
    assert parse_front_matter(text) == ({}, text)


def test_text_document_gets_metadata(tmp_path):
    path = write(tmp_path / "doc.md", GOOD)
    document = load_text_document(path, "doc.md")
    assert (document.access, document.department, document.updated, document.title) == (
        "managers", "Finance", "2026-03-01", "My Doc",
    )
    assert document.segments[0].page is None


def test_windows_line_endings_and_bom_are_handled(tmp_path):
    path = tmp_path / "doc.md"
    path.write_bytes(b"\xef\xbb\xbf" + GOOD.replace("\n", "\r\n").encode("utf-8"))
    document = load_text_document(path, "doc.md")
    assert document.access == "managers"
    assert "\r" not in document.segments[0].text


def test_a_repeated_access_key_is_rejected_instead_of_taking_the_last_value(tmp_path):
    """'access: executive' followed by 'access: all' must NOT quietly become 'all'."""
    path = write(tmp_path / "secret.md", "---\naccess: executive\ntitle: T\naccess: all\n---\nBody")
    with pytest.raises(ValueError) as excinfo:
        load_text_document(path, "secret.md")
    message = str(excinfo.value)
    assert "secret.md" in message and "'access'" in message and "more than once" in message


@pytest.mark.parametrize(
    "front_matter, key",
    [
        ("access: all\ntitle: One\ntitle: Two", "title"),
        ("access: all\nupdated: 2026-01-01\nupdated: 2026-02-02", "updated"),
        ("access: all\ndepartment: HR\ndepartment: HR", "department"),  # even an identical repeat
        ("access: all\nAccess: managers", "access"),  # keys are case-insensitive
        ("access: all\n  access : all", "access"),  # and spacing does not hide a repeat
    ],
)
def test_any_repeated_front_matter_key_is_rejected(tmp_path, front_matter, key):
    path = write(tmp_path / "doc.md", f"---\n{front_matter}\n---\nBody")
    with pytest.raises(ValueError, match=f"doc.md.*'{key}'.*more than once"):
        load_text_document(path, "doc.md")


def test_a_repeated_key_in_one_file_stops_load_documents_and_names_that_file(tmp_path):
    write(tmp_path / "good.md", GOOD)
    (tmp_path / "sub").mkdir()
    write(tmp_path / "sub" / "bad.md", "---\naccess: all\naccess: executive\n---\nBody")
    with pytest.raises(ValueError, match=r"sub/bad\.md.*'access'"):
        load_documents(tmp_path)


def test_parse_front_matter_names_the_source_it_was_given():
    with pytest.raises(ValueError, match="my-file.md"):
        parse_front_matter("---\naccess: all\naccess: all\n---\nx", "my-file.md")


def test_a_key_that_appears_once_is_fine_and_a_repeat_in_the_body_is_ignored():
    metadata, body = parse_front_matter("---\naccess: all\n---\naccess: executive\naccess: all\n")
    assert metadata == {"access": "all"}
    assert "access: executive" in body


def test_missing_access_level_is_rejected(tmp_path):
    path = write(tmp_path / "doc.md", "---\ntitle: X\n---\nBody")
    with pytest.raises(ValueError, match="access"):
        load_text_document(path, "doc.md")


def test_document_with_no_front_matter_at_all_is_rejected(tmp_path):
    path = write(tmp_path / "doc.md", "# Heading\nBody")
    with pytest.raises(ValueError, match="access"):
        load_text_document(path, "doc.md")


def test_unknown_access_level_is_rejected(tmp_path):
    path = write(tmp_path / "doc.md", "---\naccess: everyone\n---\nBody")
    with pytest.raises(ValueError, match="unknown access level"):
        load_text_document(path, "doc.md")


def test_bad_date_is_rejected(tmp_path):
    path = write(tmp_path / "doc.md", "---\naccess: all\nupdated: last week\n---\nBody")
    with pytest.raises(ValueError, match="ISO date"):
        load_text_document(path, "doc.md")


def test_shipped_documents_load_with_the_expected_access_levels(docs_dir):
    documents = load_documents(docs_dir)
    access = {document.source: document.access for document in documents}
    assert access == {
        "exec-compensation-2026.md": "executive",
        "expense-policy.md": "all",
        "hr-policy-2026.md": "all",
        "it-security-policy.md": "all",
    }
    assert [d.source for d in documents] == sorted(access)  # deterministic order


def test_shipped_documents_are_300_to_800_words(documents):
    for document in documents:
        words = len(document.segments[0].text.split())
        assert 300 <= words <= 800, (document.source, words)


def test_empty_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path)


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")


# ---------------------------------------------------------------------- PDF


def make_pdf(page_texts: list[str]) -> bytes:
    """Build a tiny valid PDF by hand (Helvetica text only, no parentheses in the text)."""
    count = len(page_texts)
    first_page_object = 4
    kids = " ".join(f"{first_page_object + 2 * i} 0 R" for i in range(count))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(page_texts):
        content_object = first_page_object + 2 * i + 1
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_object} 0 R "
                "/Resources << /Font << /F1 3 0 R >> >> >>"
            ).encode()
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
    output = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_position = len(output)
    output += f"xref\n0 {len(objects) + 1}\n".encode()
    output += b"0000000000 65535 f \n"
    for offset in offsets:
        output += f"{offset:010d} 00000 n \n".encode()
    output += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_position}\n%%EOF\n"
    ).encode()
    return bytes(output)


@pytest.mark.skipif(not HAS_PYPDF, reason="optional dependency pypdf is not installed")
def test_pdf_pages_are_loaded_with_page_numbers(tmp_path):
    (tmp_path / "Handbook.pdf").write_bytes(make_pdf(["First page about leave", "Second page about pay"]))
    write(tmp_path / "Handbook.pdf.meta", "access: managers\nupdated: 2026-02-02\ndepartment: HR\n")
    (document,) = load_documents(tmp_path)
    assert document.source == "Handbook.pdf"
    assert document.access == "managers"
    assert [segment.page for segment in document.segments] == [1, 2]
    assert "Second page" in document.segments[1].text


@pytest.mark.skipif(not HAS_PYPDF, reason="optional dependency pypdf is not installed")
def test_pdf_without_metadata_file_is_rejected(tmp_path):
    (tmp_path / "Handbook.pdf").write_bytes(make_pdf(["Some text"]))
    with pytest.raises(ValueError, match=r"\.meta"):
        load_documents(tmp_path)


@pytest.mark.skipif(not HAS_PYPDF, reason="optional dependency pypdf is not installed")
def test_a_repeated_key_in_a_pdf_sidecar_is_rejected_and_names_the_sidecar(tmp_path):
    (tmp_path / "Handbook.pdf").write_bytes(make_pdf(["Some text"]))
    write(tmp_path / "Handbook.pdf.meta", "access: executive\naccess: all\n")
    with pytest.raises(ValueError, match=r"Handbook\.pdf\.meta.*'access'.*more than once"):
        load_documents(tmp_path)


@pytest.mark.skipif(HAS_PYPDF, reason="only meaningful when pypdf is NOT installed")
def test_pdf_without_pypdf_gives_a_helpful_error(tmp_path):
    (tmp_path / "Handbook.pdf").write_bytes(b"%PDF-1.4")
    with pytest.raises(ImportError, match="pypdf"):
        load_documents(tmp_path)
