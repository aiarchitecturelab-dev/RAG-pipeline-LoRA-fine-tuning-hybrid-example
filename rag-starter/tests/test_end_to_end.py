"""End to end: the parental-leave question, the CLIs, and a dry run without any API key."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

QUESTION = "What's our parental leave policy?"


def run_script(project_root: Path, *args: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    """Run one of the CLI scripts in a subprocess WITHOUT any Anthropic credentials."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("ANTHROPIC_") and k not in ("LLM_MODEL", "EMBEDDER")}
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, *args], cwd=project_root, env=env,
        capture_output=True, text=True, encoding="utf-8",
    )


# ------------------------------------------------------------- in-process


def test_parental_leave_question_retrieves_the_hr_policy_chunk_as_top_result(pipeline):
    prepared = pipeline.prepare(QUESTION, "employee")
    top = prepared.hits[0]
    assert top.chunk.source == "hr-policy-2026.md"
    assert top.chunk.section == "Parental leave"
    assert "12 weeks of paid parental leave" in top.chunk.text


def test_the_prompt_contains_the_parental_leave_text_and_its_citation_tag(pipeline):
    prompt = pipeline.prepare(QUESTION, "employee").prompt
    assert "12 weeks of paid parental leave" in prompt.user
    assert '[Source: hr-policy-2026.md, Parental leave]' in prompt.user
    assert QUESTION in prompt.user


def test_the_final_hits_are_limited_to_rerank_top_n(pipeline, config):
    prepared = pipeline.prepare(QUESTION, "employee")
    assert 1 <= len(prepared.hits) <= config.rerank_top_n
    scores = [hit.score for hit in prepared.hits]
    assert scores == sorted(scores, reverse=True)


def test_prepare_uses_the_defaults_only_for_none_and_rejects_zero(pipeline, config):
    """Defaults apply only when the value is None; an explicit 0 is an error, not the default."""
    with pytest.raises(ValueError, match="at least 1"):
        pipeline.prepare(QUESTION, "employee", top_k=0)
    with pytest.raises(ValueError, match="at least 1"):
        pipeline.prepare(QUESTION, "employee", top_n=0)
    with pytest.raises(ValueError, match="at least 1"):
        pipeline.prepare(QUESTION, "employee", top_n=-3)
    default = pipeline.prepare(QUESTION, "employee", top_k=None, top_n=None)
    assert 1 <= len(default.hits) <= config.rerank_top_n
    assert len(pipeline.prepare(QUESTION, "employee", top_n=1).hits) == 1


def test_reranking_keeps_the_vector_score_and_adds_a_new_score(pipeline):
    top = pipeline.prepare(QUESTION, "employee").hits[0]
    assert top.vector_score > 0
    assert top.score != top.vector_score


# ------------------------------------------------------------------ the CLIs


@pytest.fixture(scope="module")
def cli_index(tmp_path_factory, project_root) -> Path:
    index_dir = tmp_path_factory.mktemp("cli") / "index"
    result = run_script(project_root, "ingest.py", "--docs", "data/docs", "--index", str(index_dir))
    assert result.returncode == 0, result.stderr
    return index_dir


def test_ingest_reports_what_it_built(cli_index, project_root):
    result = run_script(project_root, "ingest.py", "--docs", "data/docs", "--index", str(cli_index))
    assert result.returncode == 0, result.stderr
    assert "Loaded 4 documents" in result.stdout
    assert "access=executive" in result.stdout
    assert "Embedder: hashing" in result.stdout
    assert (cli_index / "vectors.npz").is_file()
    assert (cli_index / "chunks.json").is_file()


def test_ingest_accepts_a_small_chunk_size_and_shrinks_the_overlap_to_fit(tmp_path, project_root):
    result = run_script(project_root, "ingest.py", "--docs", "data/docs",
                        "--index", str(tmp_path / "small"), "--chunk-words", "40")
    assert result.returncode == 0, result.stderr
    assert "chunk_words=40, overlap=10" in result.stdout


def test_ingest_refuses_a_document_without_an_access_level(tmp_path, project_root):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "no-access.md").write_text("# Title\n\nSome text without front matter.\n", encoding="utf-8")
    result = run_script(project_root, "ingest.py", "--docs", str(docs), "--index", str(tmp_path / "idx"))
    assert result.returncode == 1
    assert "access" in result.stderr
    assert not (tmp_path / "idx").exists()  # nothing half-built is left behind


def test_dry_run_prints_sources_and_the_exact_prompt_without_an_api_key(cli_index, project_root):
    result = run_script(project_root, "query.py", QUESTION, "--role", "employee",
                        "--dry-run", "--index", str(cli_index))
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "RETRIEVED SOURCES" in out
    assert "hr-policy-2026.md | Parental leave" in out
    assert "SYSTEM PROMPT" in out and "USER PROMPT" in out
    assert '<source id="1" cite="[Source: hr-policy-2026.md, Parental leave]"' in out
    assert "12 weeks of paid parental leave" in out
    assert "Nothing was sent" in out
    assert out.isascii(), "console output must be plain ASCII (Windows consoles)"


def test_show_chunks_prints_the_chunk_text_under_each_score(cli_index, project_root):
    result = run_script(project_root, "query.py", QUESTION, "--dry-run", "--show-chunks",
                        "--index", str(cli_index))
    assert result.returncode == 0, result.stderr
    listing = result.stdout.split("SYSTEM PROMPT")[0]  # only the source list, before the prompt
    assert "12 weeks of paid parental leave" in listing


def test_cli_employee_dry_run_never_shows_executive_text(cli_index, project_root):
    result = run_script(project_root, "query.py", "What is the CEO's base salary?",
                        "--role", "employee", "--dry-run", "--index", str(cli_index),
                        "--explain-access")
    assert result.returncode == 0, result.stderr
    assert "exec-compensation-2026.md" not in result.stdout
    assert "620,000" not in result.stdout
    assert "removed 9 chunk(s)" in result.stdout


def test_cli_executive_dry_run_shows_executive_text(cli_index, project_root):
    result = run_script(project_root, "query.py", "What is the CEO's base salary?",
                        "--role", "executive", "--dry-run", "--index", str(cli_index))
    assert result.returncode == 0, result.stderr
    assert "exec-compensation-2026.md | Base salary bands" in result.stdout
    assert "620,000" in result.stdout
    assert result.stdout.isascii()


def test_cli_without_an_index_explains_how_to_create_one(tmp_path, project_root):
    result = run_script(project_root, "query.py", QUESTION, "--dry-run",
                        "--index", str(tmp_path / "missing"))
    assert result.returncode == 1
    assert "No index found" in result.stderr
    assert "Run this from inside the rag-starter folder." in result.stderr
    assert "ingest.py" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("flag", ["--top-k", "--top-n"])
@pytest.mark.parametrize("value", ["0", "-1", "abc", "2.5"])
def test_cli_rejects_a_top_k_or_top_n_that_is_not_a_positive_integer(cli_index, project_root, flag, value):
    result = run_script(project_root, "query.py", QUESTION, "--dry-run",
                        "--index", str(cli_index), flag, value)
    assert result.returncode == 2  # argparse usage error, not a crash
    assert f"argument {flag}" in result.stderr
    assert "at least 1" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_accepts_a_positive_top_k_and_top_n_and_reports_them(cli_index, project_root):
    result = run_script(project_root, "query.py", QUESTION, "--dry-run",
                        "--index", str(cli_index), "--top-k", "3", "--top-n", "2")
    assert result.returncode == 0, result.stderr
    assert "TOP_K=3, RERANK_TOP_N=2" in result.stdout
    assert "kept 2 after re-ranking" in result.stdout


def test_cli_top_k_and_top_n_fall_back_to_the_configured_values_only_when_absent(cli_index, project_root):
    result = run_script(project_root, "query.py", QUESTION, "--dry-run", "--index", str(cli_index),
                        extra_env={"TOP_K": "7", "RERANK_TOP_N": "4"})
    assert result.returncode == 0, result.stderr
    assert "TOP_K=7, RERANK_TOP_N=4" in result.stdout


def _copy_index_with(tmp_path: Path, cli_index: Path, edit) -> Path:
    """Copy the good index and let `edit(payload)` damage its chunks.json."""
    import json
    import shutil

    broken = tmp_path / "broken-index"
    shutil.copytree(cli_index, broken)
    payload = json.loads((broken / "chunks.json").read_text(encoding="utf-8"))
    edit(payload)
    (broken / "chunks.json").write_text(json.dumps(payload), encoding="utf-8")
    return broken


@pytest.mark.parametrize(
    "damage, expected_exception",
    [
        (lambda payload: payload.pop("embedder"), "KeyError"),
        (lambda payload: payload.pop("chunks"), "KeyError"),
        (lambda payload: payload["chunks"][0].pop("access"), "TypeError"),
        (lambda payload: payload["chunks"][0].update(surprise="x"), "TypeError"),
    ],
)
def test_cli_shows_a_friendly_message_for_a_corrupt_index(
    cli_index, project_root, tmp_path, damage, expected_exception
):
    broken = _copy_index_with(tmp_path, cli_index, damage)
    result = run_script(project_root, "query.py", QUESTION, "--dry-run", "--index", str(broken))
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "looks corrupt" in result.stderr
    assert expected_exception in result.stderr
    assert "python ingest.py" in result.stderr  # says how to fix it


def test_cli_shows_a_friendly_message_when_chunks_json_is_not_json(cli_index, project_root, tmp_path):
    import shutil

    broken = tmp_path / "broken-index"
    shutil.copytree(cli_index, broken)
    (broken / "chunks.json").write_text("{ not json", encoding="utf-8")
    result = run_script(project_root, "query.py", QUESTION, "--dry-run", "--index", str(broken))
    assert result.returncode == 1
    assert "not valid JSON" in result.stderr and "Traceback" not in result.stderr


def test_cli_rejects_an_unknown_role(cli_index, project_root):
    result = run_script(project_root, "query.py", QUESTION, "--role", "intern",
                        "--dry-run", "--index", str(cli_index))
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_live_query_without_credentials_fails_cleanly(cli_index, project_root, tmp_path):
    """No key anywhere: a friendly error, exit code 1, and no traceback.

    The subprocess has every ANTHROPIC_* variable removed and its home/config
    folders pointed at an empty temporary folder, so the SDK cannot find a stored
    login profile either. The SDK then refuses to send the request, so nothing
    is sent over the network.
    """
    empty_home = str(tmp_path)
    isolated_home = {"HOME": empty_home, "USERPROFILE": empty_home, "APPDATA": empty_home,
                     "LOCALAPPDATA": empty_home, "XDG_CONFIG_HOME": empty_home}
    result = run_script(project_root, "query.py", QUESTION, "--index", str(cli_index),
                        extra_env=isolated_home)
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "ANTHROPIC_API_KEY" in result.stderr
