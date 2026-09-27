"""Shared helpers for the LoRA behavior fine-tuning starter.

This module deliberately does NOT import torch, transformers or peft, so the
data checks, the label-masking helper and the schema checker can be unit-tested
on any machine (see tests/). Tokenizers are used through duck typing only.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, NoReturn, Sequence

# The contract the fine-tuned model must follow: exactly one JSON object with
# exactly these keys.
REQUIRED_KEYS = ("answer", "tone", "sources", "needs_escalation")
EXPECTED_TONE = "formal"

# Positions with this label id are ignored by the cross-entropy loss.
IGNORE_INDEX = -100

# A deliberately generic system prompt used ONLY for the general_probe prompts.
PROBE_SYSTEM_PROMPT = "You are a helpful assistant."

# Used only by evaluate.py --prompt-baseline: the same task, but the format is
# requested in words instead of being trained into the weights.
INSTRUCTION_SYSTEM_PROMPT = (
    "You are the Globex Corp internal policy assistant. Use only the provided CONTEXT. "
    "Reply with exactly one JSON object and nothing else, with these keys: "
    '"answer" (2-3 formal sentences), "tone" (always "formal"), '
    '"sources" (list of the context ids you used) and '
    '"needs_escalation" (true if the context does not contain the answer, otherwise false). '
    "If the context does not contain the answer, say so in the answer, "
    "use an empty sources list and set needs_escalation to true."
)


# --------------------------------------------------------------------------- IO
def harden_console() -> None:
    """Make sure a stray non-ASCII character can never crash a Windows console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def exit_missing_dependency(exc: ImportError) -> NoReturn:
    """Turn a failed heavy import (torch, transformers, peft, datasets) into one readable line."""
    name = getattr(exc, "name", None) or str(exc)
    sys.exit(f"missing Python package '{name}'; install the dependencies first: pip install -r requirements.txt")


# ----------------------------------------------------------------- adapter folders
ADAPTER_CONFIG = "adapter_config.json"


def require_adapter(adapter_dir: str | Path) -> Path:
    """Exit with ONE clear line unless <adapter_dir>/adapter_config.json exists. Call it before loading any model."""
    if not (Path(adapter_dir) / ADAPTER_CONFIG).is_file():
        sys.exit(f"no adapter at {adapter_dir}; run train_lora.py first (a smoke run writes to outputs/smoke)")
    return Path(adapter_dir)


def read_adapter_base_model(adapter_dir: str | Path) -> str | None:
    """The base model id or folder recorded by PEFT in adapter_config.json (None when absent)."""
    try:
        config = json.loads((Path(adapter_dir) / ADAPTER_CONFIG).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    name = config.get("base_model_name_or_path") if isinstance(config, dict) else None
    return name if isinstance(name, str) and name.strip() else None


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a UTF-8 JSONL file (blank lines skipped) with helpful errors."""
    path = Path(path)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}, line {lineno}: invalid JSON ({exc})") from exc
    return rows


# --------------------------------------------------------- prompt construction
def has_chat_template(tokenizer: Any) -> bool:
    """True when the tokenizer ships a chat template we can use."""
    return bool(getattr(tokenizer, "chat_template", None))


def split_messages(messages: Sequence[dict]) -> tuple[list[dict], str]:
    """Split a chat into (prompt messages, assistant reply text)."""
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("the last message of a training example must have role 'assistant'")
    return list(messages[:-1]), messages[-1]["content"]


def plain_prompt(prompt_messages: Sequence[dict]) -> str:
    """Fallback prompt format for tokenizers WITHOUT a chat template."""
    parts = []
    for msg in prompt_messages:
        parts.append(f"### {msg['role'].capitalize()}:\n{msg['content']}\n")
    parts.append("### Assistant:\n")
    return "\n".join(parts)


def render_prompt(tokenizer: Any, prompt_messages: Sequence[dict]) -> str:
    """Text the model sees before it starts answering (ends with the assistant header)."""
    if has_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            list(prompt_messages), tokenize=False, add_generation_prompt=True
        )
    return plain_prompt(prompt_messages)


def _encode(tokenizer: Any, text: str, add_special_tokens: bool) -> list[int]:
    enc = tokenizer(text, add_special_tokens=add_special_tokens)
    return [int(i) for i in enc["input_ids"]]


def encode_prompt(tokenizer: Any, prompt_messages: Sequence[dict]) -> list[int]:
    """Token ids for a prompt. Used by training AND by evaluate.py so both agree.

    Chat-template text already contains the special tokens it needs, so we must
    not add them a second time. The plain fallback lets the tokenizer add its
    usual start token (for example BOS).
    """
    text = render_prompt(tokenizer, prompt_messages)
    return _encode(tokenizer, text, add_special_tokens=not has_chat_template(tokenizer))


# ------------------------------------------------------ training example + mask
def build_example(tokenizer: Any, messages: Sequence[dict], max_len: int) -> dict | None:
    """Tokenize one chat and mask the loss so only the assistant turn is trained on.

    "Assistant turn" means the reply text plus whatever the chat template appends after it
    (for Qwen-style ChatML templates: the end-of-turn token and a trailing newline); for a
    tokenizer without a chat template it is the reply plus the EOS token.

    Returns {"input_ids", "attention_mask", "labels"} where labels are IGNORE_INDEX
    for every prompt token (system + user + assistant header). Returns None when
    the example is longer than max_len (we drop it rather than cut the JSON in half,
    because a truncated JSON target would teach the model to produce broken JSON).
    """
    prompt_messages, assistant_text = split_messages(messages)
    prompt_ids = encode_prompt(tokenizer, prompt_messages)

    if has_chat_template(tokenizer):
        prompt_text = render_prompt(tokenizer, prompt_messages)
        full_text = tokenizer.apply_chat_template(list(messages), tokenize=False)
        if full_text.startswith(prompt_text):
            completion_text = full_text[len(prompt_text):]
        else:  # unusual template that rewrites earlier turns: fall back to reply + EOS
            completion_text = assistant_text + (getattr(tokenizer, "eos_token", None) or "")
        completion_ids = _encode(tokenizer, completion_text, add_special_tokens=False)
    else:
        completion_ids = _encode(tokenizer, assistant_text, add_special_tokens=False)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:  # teach the model where to stop
            completion_ids = completion_ids + [int(eos_id)]

    input_ids = prompt_ids + completion_ids
    if len(input_ids) > max_len:
        return None
    labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def count_trained_tokens(example: dict) -> int:
    """Number of positions that contribute to the loss."""
    return sum(1 for x in example["labels"] if x != IGNORE_INDEX)


def pad_batch(features: Sequence[dict], pad_id: int) -> dict[str, list[list[int]]]:
    """Right-pad a list of examples (pure Python; the trainer wraps it in tensors)."""
    longest = max(len(f["input_ids"]) for f in features)
    batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        gap = longest - len(f["input_ids"])
        batch["input_ids"].append(list(f["input_ids"]) + [pad_id] * gap)
        batch["attention_mask"].append(list(f["attention_mask"]) + [0] * gap)
        batch["labels"].append(list(f["labels"]) + [IGNORE_INDEX] * gap)
    return batch


# ----------------------------------------------------------- output validation
def check_schema(obj: Any) -> list[str]:
    """Return a list of problems; an empty list means the object is schema-compliant.

    Contract: a JSON object with exactly the keys answer (non-empty str),
    tone (the str "formal"), sources (list of non-empty str) and
    needs_escalation (bool).
    """
    if not isinstance(obj, dict):
        return ["not a JSON object"]
    problems: list[str] = []
    missing = [k for k in REQUIRED_KEYS if k not in obj]
    extra = [k for k in obj if k not in REQUIRED_KEYS]
    if missing:
        problems.append("missing keys: " + ", ".join(missing))
    if extra:
        problems.append("unexpected keys: " + ", ".join(sorted(extra)))
    if "answer" in obj and not (isinstance(obj["answer"], str) and obj["answer"].strip()):
        problems.append("answer must be a non-empty string")
    if "tone" in obj and obj["tone"] != EXPECTED_TONE:
        problems.append(f"tone must be the string '{EXPECTED_TONE}'")
    if "sources" in obj:
        src = obj["sources"]
        if not (isinstance(src, list) and all(isinstance(s, str) and s.strip() for s in src)):
            problems.append("sources must be a list of non-empty strings")
    if "needs_escalation" in obj and not isinstance(obj["needs_escalation"], bool):
        problems.append("needs_escalation must be a boolean")
    return problems


def parse_json_strict(text: str) -> dict | None:
    """The WHOLE output (ignoring surrounding whitespace) must be one JSON object."""
    try:
        obj = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def parse_json_lenient(text: str) -> dict | None:
    """Find a JSON object inside code fences or surrounding prose (informational metric)."""
    strict = parse_json_strict(text)
    if strict is not None:
        return strict
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for chunk in candidates:
        start = chunk.find("{")
        while start != -1:
            try:
                obj, _ = decoder.raw_decode(chunk[start:])
            except (json.JSONDecodeError, ValueError):
                obj = None
            if isinstance(obj, dict):
                return obj
            start = chunk.find("{", start + 1)
    return None


def probe_pass(output: str, expected: str | Iterable[str]) -> bool:
    """Did a general-knowledge answer contain the expected text?

    Case-insensitive substring match. A purely numeric expectation must not be
    part of a longer number (so "7" does not match "17").
    """
    options = [expected] if isinstance(expected, str) else list(expected)
    for opt in options:
        if opt.isdigit():
            if re.search(rf"(?<!\d){re.escape(opt)}(?!\d)", output):
                return True
        elif opt.lower() in output.lower():
            return True
    return False


# ----------------------------------------------------------------- loss history
def summarise_loss_history(log_history: Iterable[dict]) -> list[dict]:
    """Turn Trainer.state.log_history into rows of {epoch, train_loss, val_loss}."""
    rows: dict[float, dict] = {}
    for entry in log_history:
        if "epoch" not in entry:
            continue
        key = round(float(entry["epoch"]), 3)
        if "eval_loss" in entry:
            rows.setdefault(key, {"epoch": key})["val_loss"] = float(entry["eval_loss"])
        elif "loss" in entry:
            rows.setdefault(key, {"epoch": key})["train_loss"] = float(entry["loss"])
    return [rows[k] for k in sorted(rows)]


def overfitting_hint(history: Sequence[dict]) -> str:
    """One plain-English sentence about the loss curves. Not a proof of anything."""
    val = [(r["epoch"], r["val_loss"]) for r in history if "val_loss" in r]
    trn = [r["train_loss"] for r in history if "train_loss" in r]
    if len(val) < 2 or len(trn) < 2:
        return "Not enough epochs to judge overfitting (need at least 2 evaluations)."
    best_epoch, best_val = min(val, key=lambda t: t[1])
    last_val = val[-1][1]
    train_falling = trn[-1] < trn[-2]
    if last_val > best_val + 1e-9 and train_falling:
        return (
            f"Validation loss is lower at epoch {best_epoch:g} ({best_val:.4f}) than at the end "
            f"({last_val:.4f}) while training loss kept falling: a typical sign of overfitting. "
            "Consider fewer epochs, a lower learning rate, a smaller rank or more varied data."
        )
    if last_val > best_val + 1e-9:
        return (
            f"Validation loss is higher at the end ({last_val:.4f}) than at epoch {best_epoch:g} "
            f"({best_val:.4f}) but training loss did not fall in the last epoch: training may be "
            "unstable or noisy. Consider a lower learning rate."
        )
    return (
        "Validation loss did not rise after its minimum in these numbers. That is a good sign but "
        "not proof of quality: also run evaluate.py on held-out data."
    )


def training_remark(history: Sequence[dict], smoke: bool) -> str:
    """The last line train_lora.py prints about the loss curves.

    A smoke run (a few steps on a handful of examples) must not get an interpretive
    overfitting hint: its loss numbers say nothing about training quality.
    """
    if smoke:
        return "smoke run: losses are meaningless (plumbing check only)"
    return overfitting_hint(history)


# ---------------------------------------------------- transformers compatibility
def pick_supported_name(available: Iterable[str], *candidates: str) -> str:
    """Return the first candidate name found in `available`.

    Used to survive API renames between transformers versions, e.g.
    TrainingArguments(evaluation_strategy=...) -> TrainingArguments(eval_strategy=...)
    and Trainer(tokenizer=...) -> Trainer(processing_class=...).
    """
    names = set(available)
    for cand in candidates:
        if cand in names:
            return cand
    raise RuntimeError(
        "None of the expected argument names " + ", ".join(candidates) +
        " exist in the installed transformers version; please check its documentation."
    )
