"""The evaluation set and script: the labels must be true, and the script must run."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import sys

import pytest

from rag.chunking import split_sections
from rag.config import ROLES


@pytest.fixture(scope="module")
def eval_module(project_root):
    path = project_root / "evals" / "eval_retrieval.py"
    spec = importlib.util.spec_from_file_location("eval_retrieval", path)
    module = importlib.util.module_from_spec(spec)
    # Register before executing: @dataclass looks its own module up in sys.modules.
    sys.modules["eval_retrieval"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def questions(project_root):
    path = project_root / "data" / "eval_questions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def squash(text: str) -> str:
    return " ".join(text.split()).casefold()


def test_question_set_has_a_sensible_size_and_unique_ids(questions):
    assert 15 <= len(questions) <= 20
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))
    assert all(q["role"] in ROLES for q in questions)


def test_every_label_points_at_a_real_section_that_contains_the_evidence(questions, documents):
    """The eval is only meaningful if its answers are really in the documents."""
    by_source = {d.source: dict(split_sections(d.segments[0].text, d.title)) for d in documents}
    answerable = [q for q in questions if q["expected_source"] is not None]
    assert len(answerable) >= 15
    for q in answerable:
        sections = by_source[q["expected_source"]]  # KeyError = unknown file
        assert q["expected_section"] in sections, (q["id"], q["expected_section"])
        assert squash(q["evidence"]) in squash(sections[q["expected_section"]]), q["id"]


def test_expected_documents_are_readable_by_the_asking_role(questions, documents):
    access = {d.source: d.access for d in documents}
    from rag.config import ROLE_ACCESS

    for q in questions:
        if q["expected_source"] is not None:
            assert access[q["expected_source"]] in ROLE_ACCESS[q["role"]], q["id"]


def test_there_are_access_control_probes_asking_for_executive_data_as_an_employee(questions):
    probes = [q for q in questions if q["expected_source"] is None]
    assert probes
    assert all(q["role"] == "employee" for q in probes)


def test_run_eval_measures_ranks_consistently(eval_module, pipeline, questions):
    result = eval_module.run_eval(pipeline, questions, ks=(1, 3, 5))
    assert result.answerable + result.probes == len(questions)
    assert result.access_violations == 0
    for k_small, k_large in ((1, 3), (3, 5)):
        for ranks in (result.section_ranks, result.evidence_ranks):
            assert result.hit_at(ranks, k_small) <= result.hit_at(ranks, k_large) <= result.answerable
    # an evidence hit implies a section hit at the same or an earlier rank
    for section_rank, evidence_rank in zip(result.section_ranks, result.evidence_ranks):
        if evidence_rank is not None:
            assert section_rank is not None and section_rank <= evidence_rank


def test_eval_cli_accepts_chunk_words_and_reports_it(project_root):
    result = subprocess.run(
        [sys.executable, "evals/eval_retrieval.py", "--chunk-words", "60"],
        cwd=project_root, capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert "chunk_words=60" in result.stdout
    assert "section hit" in result.stdout and "evidence hit" in result.stdout
    assert "0 restricted chunk(s) returned" in result.stdout
    assert result.stdout.isascii()


def test_smaller_chunks_mean_more_chunks(eval_module, project_root, config, embedder):
    from dataclasses import replace

    counts = []
    for words in (30, 120):
        cfg = replace(config, chunk_words=words, chunk_overlap=words // 4)
        pipe = eval_module.build_pipeline(
            project_root / "data" / "docs", cfg, embedder, eval_module.make_reranker("lexical")
        )
        counts.append(len(pipe.store))
    assert counts[0] > counts[1]


def test_no_rerank_flag_switches_the_reranker_off(project_root):
    result = subprocess.run(
        [sys.executable, "evals/eval_retrieval.py", "--no-rerank"],
        cwd=project_root, capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert "re-rank: off" in result.stdout


def test_sweep_prints_one_row_per_chunk_size(project_root):
    result = subprocess.run(
        [sys.executable, "evals/eval_retrieval.py", "--sweep", "30,120"],
        cwd=project_root, capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    rows = [line for line in result.stdout.splitlines() if line.strip()[:3] in ("30 ", "120")]
    assert len(rows) == 2


def test_sweep_prints_the_access_control_violation_total_for_every_row(project_root):
    result = subprocess.run(
        [sys.executable, "evals/eval_retrieval.py", "--sweep", "30,120"],
        cwd=project_root, capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    header = next(line for line in lines if "chunk_words" in line)
    assert header.split()[-1] == "access_viol"
    rows = [line for line in lines if line.strip()[:3] in ("30 ", "120")]
    assert [row.split()[-1] for row in rows] == ["0", "0"]  # one violation count per row
    assert "access-control check: 0 restricted chunk(s) returned across all 2 sweep row(s)" in result.stdout


def _leaky_run_eval(eval_module, monkeypatch, violations_by_call):
    """Wrap run_eval so that the n-th call reports a made-up number of violations."""
    real = eval_module.run_eval
    calls = []
    # main() reads an optional .env file into os.environ; keep the test run isolated.
    monkeypatch.setattr(eval_module, "load_dotenv", lambda *args, **kwargs: None)

    def wrapper(pipeline, questions, ks):
        result = real(pipeline, questions, ks)
        index = len(calls)
        calls.append(index)
        return dataclasses.replace(result, access_violations=violations_by_call[index])

    monkeypatch.setattr(eval_module, "run_eval", wrapper)


def test_sweep_exits_with_code_2_when_any_row_returns_a_restricted_chunk(eval_module, monkeypatch, capsys):
    _leaky_run_eval(eval_module, monkeypatch, violations_by_call=[0, 3])
    code = eval_module.main(["--sweep", "60,120"])
    out = capsys.readouterr().out
    assert code == 2
    rows = [line for line in out.splitlines() if line.strip()[:3] in ("60 ", "120")]
    assert [row.split()[-1] for row in rows] == ["0", "3"]
    assert "3 restricted chunk(s) returned across all 2 sweep row(s)" in out
    assert "FAILED" in out


def test_sweep_exits_with_code_0_when_every_row_is_clean(eval_module, monkeypatch, capsys):
    _leaky_run_eval(eval_module, monkeypatch, violations_by_call=[0, 0])
    assert eval_module.main(["--sweep", "60,120"]) == 0
    assert "FAILED" not in capsys.readouterr().out


def test_the_default_report_also_exits_with_code_2_on_a_violation(eval_module, monkeypatch, capsys):
    _leaky_run_eval(eval_module, monkeypatch, violations_by_call=[2])
    assert eval_module.main([]) == 2
    assert "2 restricted chunk(s) returned" in capsys.readouterr().out


def test_an_explicit_zero_is_rejected_not_replaced_by_the_default(eval_module, config):
    with pytest.raises(ValueError, match="CHUNK_WORDS"):
        eval_module.config_for(config, 0, None, None, None)
    with pytest.raises(ValueError, match="TOP_K"):
        eval_module.config_for(config, None, None, 0, None)
    assert eval_module.config_for(config, None, None, None, None).top_k == config.top_k
