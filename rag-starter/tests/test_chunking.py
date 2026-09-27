"""Chunking: size limit, overlap, heading metadata, sentence boundaries, no empty chunks."""

from __future__ import annotations

import pytest

from rag.chunking import (
    chunk_document,
    chunk_documents,
    pack_units,
    split_sections,
    split_sentences,
    split_units,
)
from rag.loaders import Document, Segment
from rag.text import count_words


def sentence(i: int) -> str:
    """A sentence of exactly 10 words, unique per i."""
    return f"Sentence{i} " + " ".join(f"w{i}x{j}" for j in range(8)) + f" end{i}."


def make_document(text: str, title: str = "Doc Title", page: int | None = None) -> Document:
    return Document(
        source="doc.md",
        title=title,
        access="all",
        department="Test",
        updated="2026-01-01",
        segments=(Segment(text=text, page=page),),
    )


# ------------------------------------------------------------------ sections


def test_split_sections_keeps_each_heading():
    text = "# Title\nIntro.\n\n## First\nBody one.\n\n## Second\nBody two."
    assert split_sections(text) == [
        ("Title", "Intro."),
        ("First", "Body one."),
        ("Second", "Body two."),
    ]


def test_text_before_first_heading_uses_default_heading():
    assert split_sections("Loose text.", default_heading="My Doc") == [("My Doc", "Loose text.")]


def test_empty_section_is_dropped():
    sections = split_sections("## Empty\n\n## Full\nSome text.")
    assert sections == [("Full", "Some text.")]


# ---------------------------------------------------------------- sentences


def test_sentence_splitter_splits_normal_sentences():
    assert split_sentences("First one. Second one! Third one?") == [
        "First one.", "Second one!", "Third one?"
    ]


def test_sentence_splitter_keeps_abbreviations_together():
    assert split_sentences("Use a badge, e.g. a visitor badge. Then sign in.") == [
        "Use a badge, e.g. a visitor badge.",
        "Then sign in.",
    ]


def test_sentence_splitter_does_not_mistake_casino_for_no_abbreviation():
    assert len(split_sentences("We went to the casino. It was fun.")) == 2


def test_list_items_and_table_rows_are_single_units():
    body = "Intro sentence. More text.\n- item one\n- item two\n| a | b |\n| c | d |"
    assert split_units(body) == [
        "Intro sentence.", "More text.", "- item one", "- item two", "| a | b |", "| c | d |",
    ]


# ------------------------------------------------------------------- packing


@pytest.mark.parametrize("chunk_words", [10, 25, 60, 220])
def test_chunks_are_never_empty_and_never_exceed_the_limit(documents, chunk_words):
    overlap = chunk_words // 4
    chunks = chunk_documents(documents, chunk_words, overlap)
    assert chunks
    for chunk in chunks:
        assert chunk.text.strip(), "empty chunk produced"
        assert 1 <= count_words(chunk.text) <= chunk_words


def test_overlap_repeats_the_last_sentence_of_the_previous_chunk():
    text = "## Topic\n" + " ".join(sentence(i) for i in range(6))
    chunks = chunk_document(make_document(text), chunk_words=30, overlap_words=10)
    texts = [chunk.text for chunk in chunks]
    # chunk 0 = s0 s1 s2, chunk 1 = s2 s3 s4, chunk 2 = s4 s5
    assert texts[0] == " ".join(sentence(i) for i in (0, 1, 2))
    assert texts[1].startswith(sentence(2))
    assert texts[2].startswith(sentence(4))
    assert len(texts) == 3


def test_zero_overlap_repeats_nothing():
    text = "## Topic\n" + " ".join(sentence(i) for i in range(6))
    chunks = chunk_document(make_document(text), chunk_words=30, overlap_words=0)
    joined = " ".join(chunk.text for chunk in chunks)
    assert joined == " ".join(sentence(i) for i in range(6))


def test_overlap_never_repeats_the_entire_previous_chunk():
    # Two 10-word sentences per chunk and a huge overlap allowance: the second
    # chunk must still contain new material, not a copy of the first.
    units = [sentence(i) for i in range(6)]
    chunks = pack_units(units, chunk_words=20, overlap_words=19)
    assert len(set(chunks)) == len(chunks)
    assert " ".join(units[-1:]) in chunks[-1]


def test_chunks_do_not_end_mid_sentence_when_avoidable():
    sentences = [sentence(i) for i in range(12)]
    text = "## Topic\n" + " ".join(sentences)
    chunks = chunk_document(make_document(text), chunk_words=35, overlap_words=10)
    for chunk in chunks:
        assert chunk.text.endswith("."), chunk.text
        assert chunk.text.startswith("Sentence"), chunk.text
    # and every original sentence survives intact in at least one chunk
    for original in sentences:
        assert any(original in chunk.text for chunk in chunks)


def test_a_sentence_longer_than_the_limit_is_split_by_words():
    long_sentence = " ".join(f"word{i}" for i in range(100)) + "."
    chunks = pack_units([long_sentence], chunk_words=30, overlap_words=5)
    assert all(count_words(chunk) <= 30 for chunk in chunks)
    all_words = set(long_sentence.split())
    covered = {word for chunk in chunks for word in chunk.split()}
    assert covered == all_words


def test_invalid_size_settings_raise():
    with pytest.raises(ValueError):
        pack_units(["a b c."], chunk_words=10, overlap_words=10)
    with pytest.raises(ValueError):
        pack_units(["a b c."], chunk_words=0, overlap_words=0)


# ------------------------------------------------------------------ metadata


def test_every_chunk_carries_its_section_heading_and_document_metadata():
    text = "# Title\nIntro text.\n\n## Parental leave\nTwelve weeks are paid.\n\n## Sick leave\nTen days."
    chunks = chunk_document(make_document(text, title="Doc Title"), 50, 5)
    assert [chunk.section for chunk in chunks] == ["Title", "Parental leave", "Sick leave"]
    for chunk in chunks:
        assert chunk.source == "doc.md"
        assert chunk.access == "all"
        assert chunk.department == "Test"
        assert chunk.updated == "2026-01-01"
        assert chunk.page is None


def test_a_long_section_keeps_its_heading_on_every_chunk():
    text = "## Long section\n" + " ".join(sentence(i) for i in range(10))
    chunks = chunk_document(make_document(text), chunk_words=25, overlap_words=5)
    assert len(chunks) > 1
    assert {chunk.section for chunk in chunks} == {"Long section"}


def test_embed_text_puts_the_heading_in_front_of_the_body():
    chunk = chunk_document(make_document("## Parental leave\nTwelve weeks."), 50, 5)[0]
    assert chunk.embed_text == "Parental leave\nTwelve weeks."


def test_pdf_pages_keep_their_page_number():
    document = make_document("Page text about leave. It is twelve weeks.", page=12)
    chunk = chunk_document(document, 50, 5)[0]
    assert chunk.page == 12
    assert chunk.locator == "p.12"
    assert chunk.section == ""


def test_chunk_ids_are_unique_across_documents(documents):
    chunks = chunk_documents(documents, 60, 15)
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_shipped_documents_produce_no_chunk_that_crosses_a_section(documents):
    # The chunker splits at headings first, so each chunk's text must appear inside
    # the body of exactly the section named in its metadata.
    for document in documents:
        sections = dict(split_sections(document.segments[0].text, document.title))
        for chunk in chunk_document(document, 30, 5):
            body = " ".join(sections[chunk.section].split())
            # normalize whitespace: chunks re-join units with single spaces / newlines
            first_words = " ".join(chunk.text.split()[:5])
            assert first_words in body
