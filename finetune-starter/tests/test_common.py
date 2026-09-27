"""Unit tests for the helpers in common.py and the scoring in evaluate.py (no torch, no downloads)."""
import json

import pytest

from common import (
    IGNORE_INDEX,
    build_example,
    check_schema,
    count_trained_tokens,
    encode_prompt,
    has_chat_template,
    overfitting_hint,
    pad_batch,
    parse_json_lenient,
    parse_json_strict,
    pick_supported_name,
    plain_prompt,
    probe_pass,
    read_adapter_base_model,
    require_adapter,
    split_messages,
    summarise_loss_history,
)
from evaluate import score_outputs

REPLY = json.dumps(
    {"answer": "First sentence. Second sentence.", "tone": "formal", "sources": ["HR-01"], "needs_escalation": False}
)
MESSAGES = [
    {"role": "system", "content": "You are a test assistant."},
    {"role": "user", "content": "CONTEXT:\n[HR-01] Text.\n\nQUESTION: Why?"},
    {"role": "assistant", "content": REPLY},
]


class FakeTokenizer:
    """Character-level tokenizer WITHOUT a chat template (plain fallback path)."""

    chat_template = None
    eos_token = "</s>"
    eos_token_id = 1
    bos_token_id = 2

    def __call__(self, text, add_special_tokens=True):
        ids = [ord(c) for c in text]
        return {"input_ids": ([self.bos_token_id] if add_special_tokens else []) + ids}

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i > 2)


class FakeChatTokenizer(FakeTokenizer):
    """Same, but with a ChatML-like template like real instruct models have."""

    chat_template = "fake-template"
    eos_token = "<|end|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        text = "".join(f"<|{m['role']}|>{m['content']}<|end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|assistant|>"
        return text


# ------------------------------------------------------------------- masking
def test_masking_plain_fallback_trains_only_on_reply_and_eos():
    tok = FakeTokenizer()
    assert not has_chat_template(tok)
    ex = build_example(tok, MESSAGES, max_len=2000)
    prompt_ids = encode_prompt(tok, MESSAGES[:-1])
    n = len(prompt_ids)

    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])
    assert ex["labels"][:n] == [IGNORE_INDEX] * n, "prompt tokens must be masked"
    assert ex["input_ids"][:n] == prompt_ids
    trained = ex["labels"][n:]
    assert IGNORE_INDEX not in trained
    assert tok.decode(trained[:-1]) == REPLY
    assert trained[-1] == tok.eos_token_id, "the reply must end with EOS so the model learns to stop"
    assert count_trained_tokens(ex) == len(trained)
    assert prompt_ids[0] == tok.bos_token_id, "plain fallback keeps the tokenizer's start token"


def test_masking_with_chat_template_masks_system_user_and_assistant_header():
    tok = FakeChatTokenizer()
    assert has_chat_template(tok)
    ex = build_example(tok, MESSAGES, max_len=2000)
    prompt_text = tok.apply_chat_template(MESSAGES[:-1], tokenize=False, add_generation_prompt=True)
    n = len(prompt_text)  # char-level: one id per character, no BOS added for chat templates

    assert ex["labels"][:n] == [IGNORE_INDEX] * n
    assert tok.decode(ex["input_ids"][:n]) == prompt_text
    assert tok.decode(ex["labels"][n:]) == REPLY + "<|end|>\n"
    assert prompt_text.endswith("<|assistant|>")
    assert "You are a test assistant." in tok.decode(ex["input_ids"])
    # nothing from the system or user turn is ever a training target
    trained_text = tok.decode([x for x in ex["labels"] if x != IGNORE_INDEX])
    assert "CONTEXT" not in trained_text and "test assistant" not in trained_text


def test_too_long_examples_are_dropped_not_truncated():
    tok = FakeTokenizer()
    full = build_example(tok, MESSAGES, max_len=10_000)
    assert build_example(tok, MESSAGES, max_len=len(full["input_ids"])) is not None
    assert build_example(tok, MESSAGES, max_len=len(full["input_ids"]) - 1) is None


def test_split_messages_requires_assistant_last():
    prompt, reply = split_messages(MESSAGES)
    assert len(prompt) == 2 and reply == REPLY
    with pytest.raises(ValueError):
        split_messages(MESSAGES[:-1])


def test_plain_prompt_ends_with_assistant_header():
    assert plain_prompt(MESSAGES[:-1]).endswith("### Assistant:\n")


def test_pad_batch_pads_right_and_masks_padding():
    feats = [
        {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [-100, 6, 7]},
        {"input_ids": [8], "attention_mask": [1], "labels": [8]},
    ]
    batch = pad_batch(feats, pad_id=0)
    assert batch["input_ids"] == [[5, 6, 7], [8, 0, 0]]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"] == [[-100, 6, 7], [8, -100, -100]]


# -------------------------------------------------------------- schema checker
GOOD = {"answer": "Yes.", "tone": "formal", "sources": ["HR-01"], "needs_escalation": False}


def test_schema_accepts_valid_object_and_empty_sources():
    assert check_schema(GOOD) == []
    assert check_schema({**GOOD, "sources": [], "needs_escalation": True}) == []


@pytest.mark.parametrize(
    "mutation, fragment",
    [
        ({"answer": None}, "answer"),
        ({"answer": "   "}, "answer"),
        ({"tone": "casual"}, "tone"),
        ({"sources": "HR-01"}, "sources"),
        ({"sources": ["HR-01", 3]}, "sources"),
        ({"needs_escalation": "false"}, "needs_escalation"),
        ({"needs_escalation": 0}, "needs_escalation"),
        ({"extra": 1}, "unexpected keys"),
    ],
)
def test_schema_rejects_bad_values(mutation, fragment):
    problems = check_schema({**GOOD, **mutation})
    assert problems and any(fragment in p for p in problems)


def test_schema_reports_missing_keys_and_non_objects():
    incomplete = {k: v for k, v in GOOD.items() if k != "tone"}
    assert any("missing keys: tone" in p for p in check_schema(incomplete))
    assert check_schema([GOOD]) == ["not a JSON object"]
    assert check_schema("text") == ["not a JSON object"]


# ---------------------------------------------------------------- JSON parsing
def test_strict_parser_accepts_only_a_bare_json_object():
    assert parse_json_strict(REPLY) == json.loads(REPLY)
    assert parse_json_strict("  " + REPLY + "\n") == json.loads(REPLY)
    assert parse_json_strict("Sure! " + REPLY) is None
    assert parse_json_strict("```json\n" + REPLY + "\n```") is None
    assert parse_json_strict("[1, 2]") is None
    assert parse_json_strict("") is None


def test_lenient_parser_finds_objects_in_fences_and_prose():
    assert parse_json_lenient("```json\n" + REPLY + "\n```") == json.loads(REPLY)
    assert parse_json_lenient("Here you go: " + REPLY + " Hope that helps.") == json.loads(REPLY)
    assert parse_json_lenient("no json here {oops") is None


# ----------------------------------------------------------------- probe check
def test_probe_pass_is_case_insensitive_and_numbers_need_boundaries():
    assert probe_pass("The capital is PARIS.", "Paris")
    assert not probe_pass("I do not know.", "Paris")
    assert probe_pass("There are 7 days.", ["7", "seven"])
    assert probe_pass("There are seven days.", ["7", "seven"])
    assert not probe_pass("There are 17 days.", "7")
    assert probe_pass("12 + 15 = 27", "27")


# --------------------------------------------------------------- loss history
def test_summarise_loss_history_pairs_train_and_val_by_epoch():
    log = [
        {"loss": 2.0, "epoch": 1.0},
        {"eval_loss": 1.8, "epoch": 1.0},
        {"loss": 1.0, "epoch": 2.0},
        {"eval_loss": 1.1, "epoch": 2.0},
        {"train_runtime": 5.0, "train_loss": 1.5, "epoch": 2.0},  # final summary, ignored
    ]
    rows = summarise_loss_history(log)
    assert rows == [
        {"epoch": 1.0, "train_loss": 2.0, "val_loss": 1.8},
        {"epoch": 2.0, "train_loss": 1.0, "val_loss": 1.1},
    ]


def test_overfitting_hint_detects_rising_validation_loss():
    rising = [
        {"epoch": 1, "train_loss": 2.0, "val_loss": 1.5},
        {"epoch": 2, "train_loss": 1.0, "val_loss": 1.2},
        {"epoch": 3, "train_loss": 0.4, "val_loss": 1.4},
    ]
    assert "overfitting" in overfitting_hint(rising)
    healthy = [
        {"epoch": 1, "train_loss": 2.0, "val_loss": 1.5},
        {"epoch": 2, "train_loss": 1.0, "val_loss": 1.2},
    ]
    assert "did not rise" in overfitting_hint(healthy)
    assert "Not enough" in overfitting_hint(healthy[:1])
    unstable = [
        {"epoch": 1, "train_loss": 1.0, "val_loss": 1.2},
        {"epoch": 2, "train_loss": 1.1, "val_loss": 1.5},
    ]
    hint = overfitting_hint(unstable)
    assert "did not fall" in hint and "overfitting" not in hint


# ------------------------------------------------------- version compatibility
def test_pick_supported_name_handles_both_transformers_generations():
    assert pick_supported_name({"eval_strategy", "seed"}, "eval_strategy", "evaluation_strategy") == "eval_strategy"
    assert pick_supported_name({"evaluation_strategy", "seed"}, "eval_strategy", "evaluation_strategy") == "evaluation_strategy"
    assert pick_supported_name({"tokenizer"}, "processing_class", "tokenizer") == "tokenizer"
    with pytest.raises(RuntimeError):
        pick_supported_name({"seed"}, "eval_strategy", "evaluation_strategy")


# ------------------------------------------------------------- adapter folder helpers
def test_read_adapter_base_model_handles_missing_and_broken_files(tmp_path):
    assert read_adapter_base_model(tmp_path) is None  # no adapter_config.json
    (tmp_path / "adapter_config.json").write_text("{not json", encoding="utf-8")
    assert read_adapter_base_model(tmp_path) is None
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "  "}), encoding="utf-8")
    assert read_adapter_base_model(tmp_path) is None
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "org/model"}), encoding="utf-8")
    assert read_adapter_base_model(tmp_path) == "org/model"


def test_require_adapter_accepts_a_folder_with_a_config(tmp_path):
    with pytest.raises(SystemExit, match="^no adapter at "):
        require_adapter(tmp_path)
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert require_adapter(tmp_path) == tmp_path


# --------------------------------------------------------------- evaluate.py scoring
def test_score_outputs_counts_only_what_it_measures():
    gold = [
        {"needs_escalation": False},
        {"needs_escalation": False},
        {"needs_escalation": True},
        {"needs_escalation": True},
    ]
    escalated = json.dumps({**GOOD, "sources": [], "needs_escalation": True})
    outputs = [
        json.dumps(GOOD),                      # valid + compliant, correctly not escalated
        "```json\n" + json.dumps(GOOD) + "\n```",  # only leniently valid
        escalated,                             # correct escalation
        json.dumps({**GOOD, "tone": "casual", "needs_escalation": False}),  # valid JSON, wrong tone, missed escalation
    ]
    m = score_outputs(outputs, gold, ["Paris", "no idea"], ["Paris", "Tokyo"])
    assert m["n_val"] == 4 and m["n_unanswerable"] == 2 and m["n_answerable"] == 2
    assert m["json_valid_rate"] == 0.75
    assert m["json_lenient_rate"] == 1.0
    assert m["schema_compliance_rate"] == 0.5
    assert m["escalation_accuracy_on_unanswerable"] == 0.5
    assert m["false_escalation_rate_on_answerable"] == 0.0
    assert m["general_probe_pass_rate"] == 0.5
    assert m["probe_outputs_that_are_json_rate"] == 0.0


def test_score_outputs_without_probe_or_unanswerable_examples():
    m = score_outputs([json.dumps(GOOD)], [{"needs_escalation": False}])
    assert m["escalation_accuracy_on_unanswerable"] is None
    assert m["general_probe_pass_rate"] is None
