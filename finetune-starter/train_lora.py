#!/usr/bin/env python
"""LoRA fine-tuning for BEHAVIOR (formal tone + strict JSON), not for facts.

Uses transformers + peft + datasets only (no TRL). The loss is masked so the model
is trained ONLY on the assistant reply (the JSON object); the system and user
turns are context, not training targets.

Quick start (PowerShell or bash, from the finetune-starter folder):

    python train_lora.py --smoke --model hf-internal-testing/tiny-random-LlamaForCausalLM --out-dir outputs/smoke
    python train_lora.py                              # full run with the default 0.5B model
    python train_lora.py --epochs 6 --lr 1e-4         # try other hyper-parameters

(--smoke without --model would download the default 0.5B model.)

Only the small LoRA adapter is saved (adapter_config.json + adapter_model.safetensors).
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path

from common import (
    build_example,
    count_trained_tokens,
    exit_missing_dependency,
    harden_console,
    pad_batch,
    pick_supported_name,
    read_jsonl,
    summarise_loss_history,
    training_remark,
)

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LoRA fine-tune a small causal LM to answer in a formal tone as strict JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or local folder")
    p.add_argument("--data-dir", default=str(HERE / "data"), help="folder with train.jsonl and val.jsonl")
    p.add_argument("--out-dir", default=str(HERE / "outputs" / "lora-adapter"), help="where the adapter is saved")
    p.add_argument("--epochs", type=float, default=4, help="number of passes over the training data")
    p.add_argument("--lr", type=float, default=2e-4, help="peak learning rate (LoRA usually tolerates 1e-4 to 3e-4)")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank r")
    p.add_argument("--alpha", type=float, default=32, help="LoRA alpha (scaling = alpha / rank)")
    p.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--batch-size", type=int, default=4, help="per-device batch size")
    p.add_argument("--grad-accum", type=int, default=2, help="gradient accumulation steps")
    p.add_argument("--max-len", type=int, default=512, help="drop examples longer than this many tokens")
    p.add_argument("--max-steps", type=int, default=-1, help="stop after this many optimizer steps (-1 = use --epochs)")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="quick plumbing check: only 8 train + 4 val examples and 3 optimizer steps (or --max-steps N); "
        "the losses it prints are meaningless",
    )
    return p


def load_split(tokenizer, path: Path, max_len: int, limit: int | None = None):
    rows = read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    examples, dropped = [], 0
    for row in rows:
        ex = build_example(tokenizer, row["messages"], max_len)
        if ex is None:
            dropped += 1
        else:
            examples.append(ex)
    if dropped:
        print(f"WARNING: dropped {dropped} example(s) from {path.name} because they exceed --max-len {max_len}")
    if not examples:
        raise SystemExit(f"No usable examples in {path}. Increase --max-len or check the data.")
    return examples


class Collator:
    """Right-pads a batch and turns it into tensors."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch

        keep = ("input_ids", "attention_mask", "labels")
        batch = pad_batch([{k: f[k] for k in keep} for f in features], self.pad_id)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def main(argv=None) -> int:
    harden_console()
    args = build_parser().parse_args(argv)

    # Heavy imports live here so that `python train_lora.py --help` and the unit
    # tests work without torch installed.
    try:
        import torch
        import transformers
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        exit_missing_dependency(exc)

    if args.smoke:
        args.epochs = 1
        if args.max_steps < 0:
            args.max_steps = 3
    train_limit, val_limit = (8, 4) if args.smoke else (None, None)

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    set_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    print(f"transformers {transformers.__version__} | torch {torch.__version__} | device: {device_name}")
    if not use_cuda:
        print("No GPU found: training on CPU. This works for a 0.5B model but is slow.")

    # ------------------------------------------------------------- tokenizer + data
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

    train_examples = load_split(tokenizer, data_dir / "train.jsonl", args.max_len, train_limit)
    val_examples = load_split(tokenizer, data_dir / "val.jsonl", args.max_len, val_limit)
    train_ds, val_ds = Dataset.from_list(train_examples), Dataset.from_list(val_examples)
    first = train_examples[0]
    print(
        f"train examples: {len(train_ds)} | val examples: {len(val_ds)} | first example: "
        f"{len(first['input_ids'])} tokens, {count_trained_tokens(first)} of them are assistant tokens (trained on)"
    )

    # ------------------------------------------------------------------ model + LoRA
    # Since transformers v5 the default load dtype follows the checkpoint (often bf16);
    # v4 defaulted to float32. We cast to float32 so behavior is the same on every
    # version and LoRA training is numerically stable. Mixed precision is added below.
    model = AutoModelForCausalLM.from_pretrained(args.model).float()
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules="all-linear",  # every linear layer except the output head
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # -------------------------------------------------------------- trainer set-up
    steps_per_epoch = max(1, math.ceil(len(train_ds) / (args.batch_size * args.grad_accum)))
    total_steps = args.max_steps if args.max_steps > 0 else max(1, math.ceil(steps_per_epoch * args.epochs))
    warmup_steps = max(1, round(0.1 * total_steps))
    # bf16 needs Ampere or newer; a free Colab T4 uses fp16 mixed precision instead.
    bf16 = use_cuda and torch.cuda.get_device_capability(0)[0] >= 8
    fp16 = use_cuda and not bf16

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    ta_kwargs = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.0,
        logging_strategy="epoch",  # one train-loss number per epoch ...
        save_strategy="no",  # ... and no checkpoints: only the adapter is saved at the end
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        bf16=bf16,
        fp16=fp16,
        dataloader_pin_memory=use_cuda,
        remove_unused_columns=False,
        # PEFT hides the base model's signature, so name the label column explicitly;
        # without this some versions cannot compute eval_loss.
        label_names=["labels"],
    )
    # transformers renamed evaluation_strategy -> eval_strategy; support either.
    ta_kwargs[pick_supported_name(ta_params, "eval_strategy", "evaluation_strategy")] = "epoch"
    targs = TrainingArguments(**ta_kwargs)

    class EpochPrinter(TrainerCallback):
        """Prints one line per epoch for train loss and validation loss."""

        def on_log(self, a, state, control, logs=None, **kwargs):
            if not logs:
                return
            epoch = logs.get("epoch", float("nan"))
            if "eval_loss" in logs:
                print(f"[val  ] epoch {epoch:5.2f} | val_loss   = {logs['eval_loss']:.4f}")
            elif "loss" in logs:
                print(f"[train] epoch {epoch:5.2f} | train_loss = {logs['loss']:.4f}")

    tr_params = inspect.signature(Trainer.__init__).parameters
    tr_kwargs = dict(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(pad_id),
        callbacks=[EpochPrinter()],
    )
    # transformers renamed Trainer(tokenizer=...) -> Trainer(processing_class=...); support either.
    tr_kwargs[pick_supported_name(tr_params, "processing_class", "tokenizer")] = tokenizer
    trainer = Trainer(**tr_kwargs)

    print(
        f"Training: epochs={args.epochs:g} max_steps={args.max_steps} lr={args.lr:g} rank={args.rank} "
        f"alpha={args.alpha:g} dropout={args.dropout:g} effective_batch={args.batch_size * args.grad_accum} "
        f"warmup_steps={warmup_steps} precision={'bf16' if bf16 else 'fp16' if fp16 else 'fp32'}"
    )
    trainer.train()

    # ------------------------------------------------------------------- report
    history = summarise_loss_history(trainer.state.log_history)
    print("\nLoss per epoch (train loss = mean during the epoch with dropout on; val loss = measured at epoch end)")
    print("epoch | train_loss | val_loss")
    for row in history:
        tr = f"{row['train_loss']:.4f}" if "train_loss" in row else "  n/a "
        va = f"{row['val_loss']:.4f}" if "val_loss" in row else "  n/a "
        print(f"{row['epoch']:5.2f} |   {tr}   |  {va}")
    print(training_remark(history, args.smoke))  # no overfitting reading for a smoke run

    # ------------------------------------------------------------------ save adapter
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))  # PEFT saves the adapter weights + config only
    summary = {
        "base_model": args.model,
        "hyperparameters": {
            "epochs": args.epochs, "max_steps": args.max_steps, "lr": args.lr, "rank": args.rank,
            "alpha": args.alpha, "dropout": args.dropout, "batch_size": args.batch_size,
            "grad_accum": args.grad_accum, "max_len": args.max_len, "seed": args.seed,
            "smoke": args.smoke,
        },
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "loss_history": history,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Adapter saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
