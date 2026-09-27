#!/usr/bin/env python
"""Compare the BASE model with the LoRA-TUNED model on held-out data.

Run it after train_lora.py:

    python evaluate.py --adapter outputs/lora-adapter

The base model is read from the adapter's adapter_config.json, so --model is only
needed to override it. Without --adapter only the base model is scored
(default Qwen/Qwen2.5-0.5B-Instruct).

It generates answers for every prompt in data/val.jsonl (the assistant turn is
removed and used only as the gold label) and for data/general_probe.jsonl, then
reports, for base and tuned side by side:

  (a) json_valid_rate         the whole output is exactly one JSON object
  (b) schema_compliance_rate  ... with exactly the required keys and types
  (c) escalation accuracy     on the unanswerable examples: needs_escalation must be true
  (d) general_probe_pass_rate simple general-knowledge prompts (forgetting check)

Everything is written to results.json. The script reports only what it measures:
greedy decoding, one run, a small validation set. Treat small differences as noise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import (
    INSTRUCTION_SYSTEM_PROMPT,
    PROBE_SYSTEM_PROMPT,
    check_schema,
    encode_prompt,
    exit_missing_dependency,
    harden_console,
    parse_json_lenient,
    parse_json_strict,
    probe_pass,
    read_adapter_base_model,
    read_jsonl,
    require_adapter,
    split_messages,
)

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Like ArgumentDefaultsHelpFormatter, but does not print '(default: None)'."""

    def _get_help_string(self, action):
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate base vs LoRA-tuned model: JSON validity, schema, escalation, forgetting.",
        formatter_class=HelpFormatter,
    )
    p.add_argument(
        "--model",
        default=None,
        help="base model id or local folder. Omit it with --adapter to use the base model recorded in the adapter; "
        f"without an adapter the default is {DEFAULT_MODEL}",
    )
    p.add_argument("--adapter", default=None, help="folder written by train_lora.py; omit to score only the base model")
    p.add_argument("--data-dir", default=str(HERE / "data"), help="folder with val.jsonl and general_probe.jsonl")
    p.add_argument("--out", default=str(HERE / "results.json"), help="where to write the results")
    p.add_argument("--max-new-tokens", type=int, default=220, help="generation limit for val prompts")
    p.add_argument("--probe-max-new-tokens", type=int, default=64, help="generation limit for general_probe prompts")
    p.add_argument("--batch-size", type=int, default=8, help="generation batch size")
    p.add_argument("--limit", type=int, default=None, help="only use the first N val and probe prompts (quick checks)")
    p.add_argument(
        "--prompt-baseline",
        action="store_true",
        help="also score the BASE model when the JSON format is requested in the system prompt "
        "(shows how far prompting alone gets you)",
    )
    return p


def resolve_base_model(model_arg: str | None, adapter_dir: str | None) -> tuple[str, list[str]]:
    """Pick the base model to load. Returns (model name, warnings). Pure Python, unit-testable.

    With an adapter and no --model, the name recorded in the adapter's adapter_config.json is used
    (the adapter only makes sense on the model it was trained on). An explicit --model that differs
    from the recorded one is honored but produces a warning. Without an adapter the default is used.
    """
    if adapter_dir is None:
        return (model_arg or DEFAULT_MODEL), []
    recorded = read_adapter_base_model(adapter_dir)
    if model_arg is None:
        if recorded is None:
            raise SystemExit(f"{adapter_dir}/adapter_config.json does not record a base model; pass --model")
        return recorded, []
    warnings = []
    if recorded is not None and model_arg != recorded:
        warnings.append(
            f"WARNING: --model {model_arg} differs from the base model recorded in the adapter ({recorded}). "
            "The adapter was trained on the recorded model; results may be meaningless or loading may fail."
        )
    return model_arg, warnings


# ----------------------------------------------------------------------- scoring
def _rate(num: int, den: int):
    return None if den == 0 else round(num / den, 4)


def score_outputs(val_outputs, gold_objs, probe_outputs=None, probe_expected=None) -> dict:
    """Turn raw generations into the metrics above. Pure Python, unit-testable."""
    n = len(val_outputs)
    strict = [parse_json_strict(o) for o in val_outputs]
    lenient = [parse_json_lenient(o) for o in val_outputs]
    unanswerable = [i for i, g in enumerate(gold_objs) if g.get("needs_escalation") is True]
    unanswerable_set = set(unanswerable)
    answerable = [i for i in range(n) if i not in unanswerable_set]

    def flagged(i: int) -> bool:
        return strict[i] is not None and strict[i].get("needs_escalation") is True

    counts = {
        "json_valid": sum(s is not None for s in strict),
        "json_found_leniently": sum(s is not None for s in lenient),
        "schema_compliant": sum(s is not None and not check_schema(s) for s in strict),
        "escalation_correct_on_unanswerable": sum(flagged(i) for i in unanswerable),
        "false_escalation_on_answerable": sum(flagged(i) for i in answerable),
    }
    metrics = {
        "n_val": n,
        "n_unanswerable": len(unanswerable),
        "n_answerable": len(answerable),
        "json_valid_rate": _rate(counts["json_valid"], n),
        "json_lenient_rate": _rate(counts["json_found_leniently"], n),
        "schema_compliance_rate": _rate(counts["schema_compliant"], n),
        "escalation_accuracy_on_unanswerable": _rate(counts["escalation_correct_on_unanswerable"], len(unanswerable)),
        "false_escalation_rate_on_answerable": _rate(counts["false_escalation_on_answerable"], len(answerable)),
        "counts": counts,
    }
    if probe_outputs is not None:
        passed = sum(probe_pass(o, e) for o, e in zip(probe_outputs, probe_expected))
        as_json = sum(parse_json_strict(o) is not None for o in probe_outputs)
        metrics.update(
            {
                "n_probe": len(probe_outputs),
                "general_probe_pass_rate": _rate(passed, len(probe_outputs)),
                "probe_outputs_that_are_json_rate": _rate(as_json, len(probe_outputs)),
            }
        )
        metrics["counts"].update({"probe_passed": passed, "probe_outputs_that_are_json": as_json})
    else:
        metrics.update({"n_probe": None, "general_probe_pass_rate": None, "probe_outputs_that_are_json_rate": None})
    return metrics


def fmt(value) -> str:
    return "n/a" if value is None else f"{100 * value:5.1f}%"


def print_table(columns: dict[str, dict]) -> None:
    rows = [
        ("(a) JSON valid (strict)", "json_valid_rate"),
        ("    JSON found leniently", "json_lenient_rate"),
        ("(b) schema compliant", "schema_compliance_rate"),
        ("(c) escalation ok (unanswerable)", "escalation_accuracy_on_unanswerable"),
        ("    false escalation (answerable)", "false_escalation_rate_on_answerable"),
        ("(d) general probe pass", "general_probe_pass_rate"),
        ("    probe outputs that are JSON", "probe_outputs_that_are_json_rate"),
    ]
    names = list(columns)
    width = max(len(n) for n in names + ["metric"]) + 2
    print("\n" + "metric".ljust(36) + "".join(n.rjust(width) for n in names))
    print("-" * (36 + width * len(names)))
    for label, key in rows:
        print(label.ljust(36) + "".join(fmt(columns[n].get(key)).rjust(width) for n in names))
    first = next(iter(columns.values()))
    print(
        f"\nn_val={first['n_val']} (unanswerable={first['n_unanswerable']}, answerable={first['n_answerable']}), "
        f"n_probe={first['n_probe']}. Rates on such small sets move in big steps: treat them as a smoke signal."
    )


# -------------------------------------------------------------------- generation
def generate_texts(model, tokenizer, prompt_ids: list[list[int]], max_new_tokens: int, batch_size: int, device: str) -> list[str]:
    """Greedy decoding with manual left padding; returns only the newly generated text."""
    import torch

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    order = sorted(range(len(prompt_ids)), key=lambda i: len(prompt_ids[i]))
    outputs: dict[int, str] = {}
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        longest = max(len(prompt_ids[i]) for i in idx)
        input_ids = torch.full((len(idx), longest), pad_id, dtype=torch.long)
        attention = torch.zeros((len(idx), longest), dtype=torch.long)
        for row, i in enumerate(idx):
            ids = prompt_ids[i]
            input_ids[row, longest - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attention[row, longest - len(ids):] = 1
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                # clear sampling defaults some checkpoints ship in generation_config.json
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
                pad_token_id=pad_id,
            )
        new_tokens = generated[:, longest:].cpu()
        for row, i in enumerate(idx):
            outputs[i] = tokenizer.decode(new_tokens[row], skip_special_tokens=True)
        print(f"    generated {len(outputs)}/{len(prompt_ids)}", flush=True)
    return [outputs[i] for i in range(len(prompt_ids))]


def main(argv=None) -> int:
    harden_console()
    args = build_parser().parse_args(argv)

    # Cheap checks first, before any heavy import or model load.
    if args.adapter:
        require_adapter(args.adapter)
    model_name, warnings = resolve_base_model(args.model, args.adapter)
    for line in warnings:
        print(line, file=sys.stderr)

    data_dir = Path(args.data_dir)
    val_rows = read_jsonl(data_dir / "val.jsonl")
    probe_rows = read_jsonl(data_dir / "general_probe.jsonl")
    if args.limit is not None:
        val_rows, probe_rows = val_rows[:args.limit], probe_rows[:args.limit]

    try:
        import contextlib

        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if args.adapter:
            from peft import PeftModel
    except ImportError as exc:
        exit_missing_dependency(exc)

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"transformers {transformers.__version__} | torch {torch.__version__} | device: {device}")
    print(f"base model: {model_name}" + (f" | adapter: {args.adapter}" if args.adapter else " (no adapter)"))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # float32 on every transformers version (v5 would otherwise follow the checkpoint dtype)
    model = AutoModelForCausalLM.from_pretrained(model_name).float()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device).eval()

    # Prompts: the assistant turn is removed and kept only as the gold label.
    val_prompts, gold_objs = [], []
    for row in val_rows:
        prompt_messages, gold_text = split_messages(row["messages"])
        val_prompts.append(prompt_messages)
        gold_objs.append(json.loads(gold_text))
    probe_prompts = [
        [{"role": "system", "content": PROBE_SYSTEM_PROMPT}, {"role": "user", "content": r["prompt"]}]
        for r in probe_rows
    ]
    probe_expected = [r["expected"] for r in probe_rows]

    def run(prompts, max_new_tokens):
        ids = [encode_prompt(tokenizer, m) for m in prompts]
        return generate_texts(model, tokenizer, ids, max_new_tokens, args.batch_size, device)

    def base_mode():
        # With an adapter attached, the base model = adapter temporarily disabled.
        return model.disable_adapter() if args.adapter else contextlib.nullcontext()

    print(f"Generating with the BASE model ({len(val_prompts)} val + {len(probe_prompts)} probe prompts) ...")
    with base_mode():
        base_val = run(val_prompts, args.max_new_tokens)
        base_probe = run(probe_prompts, args.probe_max_new_tokens)
        base_instr = None
        if args.prompt_baseline:
            print("Generating with the BASE model + format instructions in the system prompt ...")
            instr_prompts = [
                [{"role": "system", "content": INSTRUCTION_SYSTEM_PROMPT}] + [m for m in p if m["role"] != "system"]
                for p in val_prompts
            ]
            base_instr = run(instr_prompts, args.max_new_tokens)

    tuned_val = tuned_probe = None
    if args.adapter:
        print("Generating with the TUNED model (base + LoRA adapter) ...")
        tuned_val = run(val_prompts, args.max_new_tokens)
        tuned_probe = run(probe_prompts, args.probe_max_new_tokens)

    columns = {"base": score_outputs(base_val, gold_objs, base_probe, probe_expected)}
    if base_instr is not None:
        columns["base+prompt"] = score_outputs(base_instr, gold_objs)
    if tuned_val is not None:
        columns["tuned"] = score_outputs(tuned_val, gold_objs, tuned_probe, probe_expected)
    print_table(columns)
    if not args.adapter:
        print("\nNo --adapter given: only the base model was scored.")

    examples = []
    for i, gold in enumerate(gold_objs):
        item = {"index": i, "gold_needs_escalation": gold["needs_escalation"], "base_output": base_val[i]}
        if base_instr is not None:
            item["base_with_prompt_output"] = base_instr[i]
        if tuned_val is not None:
            item["tuned_output"] = tuned_val[i]
        examples.append(item)
    probe_items = []
    for i, row in enumerate(probe_rows):
        item = {"prompt": row["prompt"], "expected": row["expected"], "base_output": base_probe[i]}
        if tuned_probe is not None:
            item["tuned_output"] = tuned_probe[i]
        probe_items.append(item)

    results = {
        "meta": {
            "base_model": model_name,
            "adapter": args.adapter,
            "decoding": "greedy (do_sample=False), single run",
            "max_new_tokens": args.max_new_tokens,
            "probe_max_new_tokens": args.probe_max_new_tokens,
            "device": device,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "caveat": "Small evaluation sets; rates move in large steps. Measure on your own data before trusting them.",
        },
        "metrics": columns,
        "val_examples": examples,
        "probe_examples": probe_items,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
