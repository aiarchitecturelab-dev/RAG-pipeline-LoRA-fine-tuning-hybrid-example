"""Config: environment parsing and the tiny .env loader."""

from __future__ import annotations

import pytest

from rag.config import Config, load_dotenv


def test_defaults():
    config = Config.from_env({})
    assert config == Config()
    assert config.embedder == "hashing"
    assert config.llm_model == "claude-opus-5"


def test_values_are_read_from_the_environment():
    config = Config.from_env(
        {
            "EMBEDDER": "Sentence-Transformers", "CHUNK_WORDS": "300", "CHUNK_OVERLAP": "50",
            "TOP_K": "50", "RERANK_TOP_N": "3", "MIN_SCORE": "0.2", "LLM_MODEL": "claude-sonnet-5",
        }
    )
    assert config.embedder == "sentence-transformers"
    assert (config.chunk_words, config.chunk_overlap, config.top_k, config.rerank_top_n) == (300, 50, 50, 3)
    assert config.min_score == pytest.approx(0.2)
    assert config.llm_model == "claude-sonnet-5"


def test_empty_values_fall_back_to_defaults():
    """Copying .env.example and leaving a line blank must not break anything."""
    config = Config.from_env({"CHUNK_WORDS": "", "TOP_K": "  ", "LLM_MODEL": ""})
    assert config == Config()


def test_a_non_numeric_value_gives_a_clear_error():
    with pytest.raises(ValueError, match="CHUNK_WORDS must be a number"):
        Config.from_env({"CHUNK_WORDS": "lots"})


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_words": 0},
        {"chunk_words": 50, "chunk_overlap": 50},
        {"chunk_overlap": -1},
        {"top_k": 0},
        {"rerank_top_n": 0},
        {"min_score": float("nan")},
        {"min_score": float("inf")},
        {"min_score": float("-inf")},
    ],
)
def test_invalid_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        Config(**overrides)


@pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "-inf", "Infinity"])
def test_a_non_finite_min_score_in_the_environment_is_rejected(raw):
    with pytest.raises(ValueError, match="MIN_SCORE must be a finite number"):
        Config.from_env({"MIN_SCORE": raw})


@pytest.mark.parametrize("raw", ["0", "0.05", "-0.5", "1"])
def test_ordinary_min_score_values_are_accepted(raw):
    assert Config.from_env({"MIN_SCORE": raw}).min_score == float(raw)


def test_dotenv_loads_values_but_does_not_override_the_shell(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n\nEXAMPLE_ONE=hello\nEXAMPLE_TWO='quoted value'\nEXAMPLE_THREE=from-file\n"
        "EXAMPLE_EMPTY=\nnot a valid line\n",
        encoding="utf-8",
    )
    for name in ("EXAMPLE_ONE", "EXAMPLE_TWO", "EXAMPLE_EMPTY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EXAMPLE_THREE", "from-shell")

    load_dotenv(env_file)

    import os
    assert os.environ["EXAMPLE_ONE"] == "hello"
    assert os.environ["EXAMPLE_TWO"] == "quoted value"
    assert os.environ["EXAMPLE_THREE"] == "from-shell"
    assert os.environ.get("EXAMPLE_EMPTY", "") == ""
    for name in ("EXAMPLE_ONE", "EXAMPLE_TWO", "EXAMPLE_EMPTY"):
        monkeypatch.delenv(name, raising=False)


def test_a_missing_dotenv_file_is_fine(tmp_path):
    load_dotenv(tmp_path / "does-not-exist.env")


def test_the_shipped_env_example_parses_to_the_default_config(tmp_path, monkeypatch, project_root):
    """.env.example must stay in sync with the defaults in config.py."""
    import os

    names = ["ANTHROPIC_API_KEY", "LLM_MODEL", "EMBEDDER", "SENTENCE_MODEL", "CHUNK_WORDS",
             "CHUNK_OVERLAP", "TOP_K", "RERANK_TOP_N", "MIN_SCORE"]
    for name in names:
        monkeypatch.delenv(name, raising=False)
    load_dotenv(project_root / ".env.example")
    try:
        assert Config.from_env() == Config()
        assert os.environ.get("ANTHROPIC_API_KEY", "") == ""  # no key is shipped
    finally:
        for name in names:
            os.environ.pop(name, None)
