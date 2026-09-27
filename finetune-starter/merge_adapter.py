#!/usr/bin/env python
"""Merge a LoRA adapter into its base model and save a standalone model folder.

    python merge_adapter.py --adapter outputs/lora-adapter --out-dir outputs/merged-model

The merged folder is a FULL model (about 2 GB for a 0.5B model in float32), which is
why the small adapter is normally kept separate. Merging removes the need for peft at
inference time and can slightly speed up serving; it does not change what the model knows.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import exit_missing_dependency, harden_console, read_adapter_base_model, require_adapter

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    harden_console()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True, help="folder written by train_lora.py")
    p.add_argument("--base-model", default=None, help="base model id (default: the one recorded in the adapter)")
    p.add_argument("--out-dir", default=str(HERE / "outputs" / "merged-model"), help="where to save the merged model")
    args = p.parse_args(argv)

    require_adapter(args.adapter)  # one clear line if the folder is wrong, before any heavy import
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        exit_missing_dependency(exc)

    base_name = args.base_model or read_adapter_base_model(args.adapter)
    if not base_name:
        sys.exit(f"{args.adapter}/adapter_config.json does not record a base model; pass --base-model")
    print(f"Base model: {base_name}")
    base = AutoModelForCausalLM.from_pretrained(base_name).float()  # merge in float32 for precision
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir))
    AutoTokenizer.from_pretrained(base_name).save_pretrained(str(out_dir))
    print(f"Merged model saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
