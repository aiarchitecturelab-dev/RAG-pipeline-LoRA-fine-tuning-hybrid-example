#!/usr/bin/env python
"""Hybrid example: RAG supplies the FACTS, a small LoRA adapter supplies the BEHAVIOR.

    1. RETRIEVE  rag-starter finds the chunks this caller is allowed to read
                 (permission filter first, then similarity search, then re-ranking).
    2. PROMPT    the chunks become a numbered CONTEXT block, in exactly the format
                 finetune-starter was trained on.
    3. GENERATE  base model + LoRA adapter (transformers + peft) writes ONE JSON object.
    4. VALIDATE  valid JSON? the required keys? are the cited ids ones we retrieved?
                 does the answer contradict its own needs_escalation flag?

Run it from the repository root (paths are resolved relative to this file):

    python hybrid/hybrid_example.py "What's our parental leave policy?" --role employee --dry-run
    python hybrid/hybrid_example.py "What's our parental leave policy?" --role employee --adapter finetune-starter/outputs/lora-adapter

--dry-run stops after step 2. It needs only numpy: torch, transformers and peft
are imported lazily, inside generate(), and are never touched in a dry run.

Exit codes: 0 = fine, 1 = setup error (no index, no adapter, ...), 2 = the model's
output failed validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RAG_DIR = REPO / "rag-starter"          # a package: `import rag`
FT_DIR = REPO / "finetune-starter"      # plain modules: `import common`

sys.path.insert(0, str(RAG_DIR))
# Appended, not inserted first, so an installed package that happens to be called
# "evaluate" (Hugging Face has one) is never shadowed by finetune-starter/evaluate.py.
sys.path.append(str(FT_DIR))

from common import check_schema, encode_prompt, parse_json_strict  # noqa: E402  (no torch inside)
from rag.config import ROLES, Config, allowed_access_levels, load_dotenv  # noqa: E402
from rag.console import make_console_safe  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402
from rag.prompting import format_citation  # noqa: E402
from rag.vectorstore import IndexMismatchError  # noqa: E402

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # the default of finetune-starter/train_lora.py

# Must be the same system message as in finetune-starter/data/*.jsonl, because that
# is the prompt the adapter was trained with (a test compares the two). The training
# data is about a different fictional company (Globex Corp) than the RAG documents
# (Acme Corp) on purpose; see hybrid/README.md. In your own project, train with your
# own system message and change this constant to match.
SYSTEM_PROMPT = "You are the Globex Corp internal policy assistant."

# Phrases that say "the information is not in the context". Used only to catch an
# answer that contradicts its own needs_escalation flag. A keyword screen, not a
# meaning check: it misses paraphrases and can, rarely, flag a correct answer.
ABSTENTION_PHRASES = (
    "not available in the provided context",
    "not available in the context",
    "does not mention",
    "does not contain",
    "cannot find",
    "no information",
    "not enough information",
)

RULE = "=" * 72


# ------------------------------------------------------------ 2. build the prompt
def build_messages(question: str, hits) -> tuple[list[dict], dict[str, str]]:
    """Turn retrieved hits into the chat the adapter was trained on.

    Returns (messages, citations). Each chunk gets an id SRC-01, SRC-02, ... that
    looks like the ids in the training data (HR-01, FIN-02, ...). The model is
    trained to answer with those ids in "sources"; `citations` maps every id back
    to the human-readable tag [Source: file, section] for display.
    """
    lines: list[str] = []
    citations: dict[str, str] = {}
    for number, hit in enumerate(hits, start=1):
        chunk, source_id = hit.chunk, f"SRC-{number:02d}"
        # One line per passage, like the training data. Collapsing newlines also
        # means a chunk cannot start a fake numbered line of its own.
        body = " ".join(chunk.text.split())
        heading = f"{chunk.section}: " if chunk.section else ""
        lines.append(f"[{source_id}] {heading}{body}")
        citations[source_id] = format_citation(chunk)
    user = "CONTEXT:\n" + "\n".join(lines) + "\n\nQUESTION: " + question.strip()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    return messages, citations


# --------------------------------------------------------------- 3. generate
def base_model_from_adapter(adapter_dir: Path) -> str | None:
    """The base model a PEFT adapter was trained on (recorded in adapter_config.json)."""
    try:
        config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    name = config.get("base_model_name_or_path")
    return name if isinstance(name, str) and name else None


def generate(messages: list[dict], model_name: str, adapter_dir: Path, max_new_tokens: int) -> str:
    """Base model + LoRA adapter -> the generated text.

    The heavy libraries are imported here, not at the top of the file, so that
    --dry-run works on a machine without torch. The model is loaded on every call:
    fine for an example, but keep it in memory in a real service.
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            f"error: live generation needs torch, transformers and peft ({exc}).\n"
            "       pip install -r finetune-starter/requirements.txt   (or add --dry-run)"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # float32, as in finetune-starter/evaluate.py; use a lower precision if memory is tight.
    model = AutoModelForCausalLM.from_pretrained(model_name).float()
    model = PeftModel.from_pretrained(model, str(adapter_dir)).to(device).eval()

    # encode_prompt() is the function training and evaluate.py use, so the model
    # sees the prompt rendered exactly as it did during fine-tuning.
    input_ids = torch.tensor([encode_prompt(tokenizer, messages)], device=device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: the same prompt gives the same answer
            temperature=None, top_p=None, top_k=None,  # clear sampling defaults some checkpoints ship
            repetition_penalty=1.0,
            pad_token_id=pad_id,
        )
    return tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)


# --------------------------------------------------------------- 4. validate
def validate_output(text: str, allowed_ids) -> tuple[dict | None, list[str]]:
    """Check the model's output. Returns (parsed object or None, problems); no problems = pass.

    1. the WHOLE output is one JSON object (a code fence or extra prose fails);
    2. it has exactly the required keys with the right types (common.check_schema);
    3. every id in "sources" is one that was actually retrieved and shown to the model;
    4. an answer that is not an escalation cites at least one source;
    5. an answer that is not an escalation does not say the information is unavailable
       (a keyword screen over ABSTENTION_PHRASES, so a self-contradicting answer fails).

    This proves the output is well-formed, refers to retrieved ids and does not
    contradict its own flag in the obvious ways. It does NOT prove that the answer is
    true or that the cited passage supports it.
    """
    obj = parse_json_strict(text)
    if obj is None:
        return None, ["the output is not exactly one JSON object"]
    problems = check_schema(obj)
    sources = obj.get("sources")
    if isinstance(sources, list):
        unknown = [s for s in sources if isinstance(s, str) and s not in allowed_ids]
        if unknown:
            problems.append("sources that were not retrieved: " + ", ".join(unknown))
        if not sources and obj.get("needs_escalation") is False:
            problems.append("no sources cited although needs_escalation is false")
    answer = obj.get("answer")
    if obj.get("needs_escalation") is False and isinstance(answer, str) and says_unavailable(answer):
        problems.append("inconsistent: says the information is unavailable but needs_escalation is false")
    return obj, problems


def says_unavailable(answer: str) -> bool:
    """True if the answer text contains one of ABSTENTION_PHRASES (case and spacing ignored)."""
    text = " ".join(answer.lower().replace(chr(0x2019), "'").split())  # curly apostrophe -> straight
    for short, full in (("doesn't", "does not"), ("can't", "cannot"), ("isn't", "is not")):
        text = text.replace(short, full)
    return any(phrase in text for phrase in ABSTENTION_PHRASES)


# ---------------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG retrieval + a LoRA-tuned model, with output validation.")
    p.add_argument("question", help="the question to ask (put it in quotes)")
    p.add_argument("--role", choices=ROLES, default="employee",
                   help="who is asking; decides which chunks may be retrieved (default: employee)")
    p.add_argument("--index", help="rag-starter index folder (default: rag-starter/index)")
    p.add_argument("--top-n", type=int, default=3,
                   help="chunks put into the prompt (default: 3; the adapter was trained on short contexts)")
    p.add_argument("--model", help="base model id or folder (default: the one recorded in the adapter)")
    p.add_argument("--adapter", help="LoRA adapter folder (default: finetune-starter/outputs/lora-adapter)")
    p.add_argument("--max-new-tokens", type=int, default=220, help="generation limit (default: 220)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the retrieval and the exact prompt, then stop; no model is loaded")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    make_console_safe()
    args = parse_args(argv)
    load_dotenv(RAG_DIR / ".env")  # the same settings (for example EMBEDDER) as rag-starter's own scripts
    index = Path(args.index) if args.index else RAG_DIR / "index"

    # ---- 1. retrieve (the permission filter is applied inside, before any search)
    try:
        config = Config.from_env()
        pipeline = RagPipeline.from_index(index, config)
        prepared = pipeline.prepare(args.question, args.role, top_n=args.top_n)
    except (FileNotFoundError, IndexMismatchError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if isinstance(exc, FileNotFoundError):
            print("hint : from the repository root run: python rag-starter/ingest.py "
                  "--docs rag-starter/data/docs --index rag-starter/index", file=sys.stderr)
        return 1

    messages, citations = build_messages(args.question, prepared.hits)
    allowed = ", ".join(sorted(allowed_access_levels(args.role)))
    print(f"Question : {args.question}")
    print(f"Role     : {args.role} (may read access levels: {allowed})")
    print(f"Retrieval: {prepared.candidates} candidates -> kept {len(prepared.hits)} after re-ranking")
    print()
    print(RULE)
    print("RETRIEVED SOURCES (permission filter, similarity search, re-ranking)")
    print(RULE)
    if not prepared.hits:
        print("  (no sources retrieved)")
    for (source_id, tag), hit in zip(citations.items(), prepared.hits):
        print(f"[{source_id}] score={hit.score:.3f}  {tag}")
    print()

    if args.dry_run:
        print(RULE)
        print("SYSTEM PROMPT (exact text)")
        print(RULE)
        print(messages[0]["content"])
        print()
        print(RULE)
        print("USER PROMPT (exact text)")
        print(RULE)
        print(messages[1]["content"])
        print()
        print("[dry run] No model was loaded and nothing was generated.")
        return 0

    if not prepared.hits:
        print("No allowed, relevant sources were found, so the model was not called.")
        print("In production this is the moment to escalate to a human.")
        return 0

    # ---- 3. generate
    adapter = Path(args.adapter) if args.adapter else FT_DIR / "outputs" / "lora-adapter"
    if not (adapter / "adapter_config.json").is_file():
        print(f"error: no LoRA adapter found in {adapter}. Train one first "
              "(see finetune-starter/README.md) or pass --adapter.", file=sys.stderr)
        return 1
    recorded = base_model_from_adapter(adapter)
    model_name = args.model or recorded or DEFAULT_BASE_MODEL
    if recorded and model_name != recorded:
        print(f"warning: the adapter was trained on {recorded!r} but --model is {model_name!r}; "
              "an adapter only fits the model it was trained on.", file=sys.stderr)
    print(f"Generating with {model_name} + adapter {adapter} ...", flush=True)
    text = generate(messages, model_name, adapter, args.max_new_tokens)

    # ---- 4. validate
    obj, problems = validate_output(text, citations)
    print(RULE)
    print("MODEL OUTPUT (raw)")
    print(RULE)
    print(text.strip())
    print()
    print(RULE)
    print("VALIDATION: " + ("PASS" if not problems else "FAIL"))
    print(RULE)
    for problem in problems:
        print(f"  - {problem}")
    if obj is not None and not problems:
        print(f"Answer          : {obj['answer']}")
        print(f"needs_escalation: {obj['needs_escalation']}")
        for source_id in obj["sources"]:
            print(f"  {source_id} = {citations[source_id]}")
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
