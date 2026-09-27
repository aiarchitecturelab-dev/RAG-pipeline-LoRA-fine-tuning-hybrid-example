"""ACCESS CONTROL: permissions are enforced at retrieval time, before the LLM.

The key property: a chunk the asking role may not read is never returned,
never put in the prompt, and never even scored - no matter what the question says.
"""

from __future__ import annotations

import pytest

from rag.chunking import chunk_documents
from rag.config import ROLE_ACCESS, ROLES, Config, allowed_access_levels
from rag.embedders import HashingEmbedder
from rag.loaders import Document, Segment
from rag.pipeline import RagPipeline, build_store
from rag.retrieve import permission_filter, retrieve

EXEC_SOURCE = "exec-compensation-2026.md"

EXEC_QUESTIONS = [
    "What is the CEO's target bonus?",
    "What is the CEO's base salary?",
    "How much severance does the CEO receive after a change of control?",
    "executive compensation",
    "Chief Executive Officer 620,000",
    "What is the clawback policy for executive bonus payments?",
]


@pytest.mark.parametrize("question", EXEC_QUESTIONS)
def test_employee_never_retrieves_executive_chunks(pipeline, question):
    prepared = pipeline.prepare(question, "employee", top_k=100, top_n=100)
    assert all(hit.chunk.access == "all" for hit in prepared.hits)
    assert not any(hit.chunk.source == EXEC_SOURCE for hit in prepared.hits)


def test_executive_query_for_executive_compensation_does_return_executive_chunks(pipeline):
    prepared = pipeline.prepare("What is the CEO's target bonus?", "executive")
    executive_hits = [hit for hit in prepared.hits if hit.chunk.access == "executive"]
    assert executive_hits, "executive role should be able to retrieve executive chunks"
    assert prepared.hits[0].chunk.source == EXEC_SOURCE
    assert prepared.hits[0].chunk.section == "Annual bonus"


def test_employee_prompt_contains_no_executive_text(pipeline, documents):
    executive_document = next(d for d in documents if d.source == EXEC_SOURCE)
    executive_body = executive_document.segments[0].text
    prepared = pipeline.prepare("What is the CEO's base salary and bonus?", "employee")
    full_prompt = prepared.prompt.system + "\n" + prepared.prompt.user
    for secret in ("620,000", "410,000", "80 percent of base salary", "18 months of base salary"):
        assert secret in executive_body  # sanity check: the figure really is in the secret document
        assert secret not in full_prompt


def test_copying_an_executive_chunk_verbatim_still_retrieves_nothing_for_employees(pipeline, store):
    secret_chunk = next(chunk for chunk in store.chunks if chunk.access == "executive")
    prepared = pipeline.prepare(secret_chunk.text, "employee", top_k=100, top_n=100)
    assert not any(hit.chunk.access == "executive" for hit in prepared.hits)
    executive = pipeline.prepare(secret_chunk.text, "executive")
    assert executive.hits[0].chunk.chunk_id == secret_chunk.chunk_id


def test_permission_filter_removed_the_executive_chunks_and_reports_how_many(pipeline, store):
    executive_chunks = sum(1 for chunk in store.chunks if chunk.access == "executive")
    assert executive_chunks > 0
    assert pipeline.prepare("anything", "employee").excluded_by_permissions == executive_chunks
    assert pipeline.prepare("anything", "executive").excluded_by_permissions == 0


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("question", EXEC_QUESTIONS + ["parental leave", "expense receipts"])
def test_every_hit_is_within_the_roles_access_levels(pipeline, role, question):
    prepared = pipeline.prepare(question, role, top_k=100, top_n=100)
    assert {hit.chunk.access for hit in prepared.hits} <= ROLE_ACCESS[role]


def test_unknown_role_fails_closed(pipeline, store, embedder):
    with pytest.raises(ValueError, match="Unknown role"):
        allowed_access_levels("intern")
    with pytest.raises(ValueError, match="Unknown role"):
        permission_filter("superuser")
    with pytest.raises(ValueError, match="Unknown role"):
        retrieve("parental leave", store, embedder, role="", top_k=5)
    with pytest.raises(ValueError, match="Unknown role"):
        pipeline.prepare("parental leave", "Employee ")  # near-miss spelling must not pass


def test_role_hierarchy_is_strictly_widening():
    assert ROLE_ACCESS["employee"] < ROLE_ACCESS["manager"] < ROLE_ACCESS["executive"]


def test_manager_tier_sits_between_employee_and_executive():
    """Build three tiny documents, one per access level, and check each role's view."""

    def document(name: str, access: str, text: str) -> Document:
        return Document(name, name, access, "", "2026-01-01", (Segment(f"## {name}\n{text}"),))

    documents = [
        document("public.md", "all", "Shared office rules for everybody about the annual budget."),
        document("managers.md", "managers", "Manager guide about the annual budget and headcount."),
        document("executive.md", "executive", "Board paper about the annual budget and strategy."),
    ]
    config = Config()
    embedder = HashingEmbedder()
    store = build_store(documents, embedder, config)
    pipeline = RagPipeline(store, embedder, config)

    def sources(role: str) -> set[str]:
        return {h.chunk.source for h in pipeline.prepare("annual budget", role, top_k=10).hits}

    assert sources("employee") == {"public.md"}
    assert sources("manager") == {"public.md", "managers.md"}
    assert sources("executive") == {"public.md", "managers.md", "executive.md"}


def test_chunks_inherit_the_access_level_of_their_document(documents):
    access_by_source = {document.source: document.access for document in documents}
    for chunk in chunk_documents(documents, 60, 10):
        assert chunk.access == access_by_source[chunk.source]


def test_eligible_chunks_depend_on_the_role_only_not_on_the_question(store, embedder):
    """Whatever the question says, results stay inside the role's own view of the index."""
    role_view = {
        chunk.chunk_id for chunk in store.chunks if chunk.access in ROLE_ACCESS["employee"]
    }
    for question in EXEC_QUESTIONS:
        result = retrieve(question, store, embedder, "employee", top_k=1000)
        assert {hit.chunk.chunk_id for hit in result.hits} <= role_view
