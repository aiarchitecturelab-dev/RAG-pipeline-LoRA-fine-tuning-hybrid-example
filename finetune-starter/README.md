# Fine-tune for behavior, not facts (LoRA starter)

A small, honest starter for the fine-tuning half of the video *RAG vs Fine-Tuning for Enterprise LLMs*.
It teaches a small open chat model **how to answer**: in a formal, compliance-approved tone, as one strict JSON object, using only the context it is given.

It does **not** try to teach the model facts. The training data is fictional and contains no facts the pipeline relies on;
every prompt brings its own context, and nothing in the training files is meant to be memorized. That is the point.

> **Status, stated plainly.** Covered by unit tests, smoke-tested with tiny models, and run end to end on CPU with the
> default model `Qwen/Qwen2.5-0.5B-Instruct` and with the smaller `HuggingFaceTB/SmolLM2-135M-Instruct`
> (numbers and an honest reading in [One real run](#one-real-run-for-orientation-only)).
> In the default-model run the JSON format was learned (24 of 24 answers strictly valid, up from 0) and abstention was
> learned on this small validation set (all 7 unanswerable questions escalated, but 3 of 17 answerable ones were escalated
> wrongly); the 135M model learned the format but mostly not the abstention (2 of 7). One seed, a tiny validation set: read the
> caveats before drawing conclusions. Not run: any GPU or Colab session (details in
> [What was and was not tested](#what-was-and-was-not-tested)).

> **Deliberately different from the RAG corpus.** The training passages describe a different fictional company
> ("Globex Corp") and cover different topics than the documents in `../rag-starter` (which are about "Acme Corp"), so
> nothing the adapter memorizes can conflict with a fact the RAG pipeline retrieves. The video's demo question,
> "What's our parental leave policy?", appears nowhere in the training or validation data, so the adapter can never memorize an
> answer to it. `tests/test_data.py` enforces this with a keyword screen over the subjects the RAG documents cover
> (a screen, not a proof) and a check that no passage title or sentence is copied from them.

## The idea: behavior, not facts

| | Retrieval (RAG) | Fine-tuning (LoRA here) |
|---|---|---|
| Good for | fresh facts, citations, per-user access control, removing data | tone, style, output format (for example strict JSON, tool-call shapes) |
| Where knowledge lives | in your documents, looked up at question time | changed model weights |
| Weak spot | needs good chunking and retrieval | facts go stale, no citations, sensitive training data is hard to remove, risk of forgetting |

So in this demo every prompt carries its own `CONTEXT` (fictional "Globex Corp" policy text with an id such as `[HR-17]`)
and a `QUESTION`. The model is trained to reply with exactly:

```json
{"answer": "According to the provided policy, the mentoring program pairs a mentee with a senior colleague from another department for 6 months. Mentor and mentee meet once a month during working hours.",
 "tone": "formal",
 "sources": ["HR-17"],
 "needs_escalation": false}
```

- `answer`: 2-3 formal sentences that use only the context.
- `tone`: always `"formal"`.
- `sources`: the context ids the answer relies on.
- `needs_escalation`: `false` normally. In about 28 percent of the training examples (20 of 71) the context does **not**
  contain the answer; then the answer says the information is not available in the provided context, `sources` is empty and
  `needs_escalation` is `true`. The intent is to teach abstention instead of guessing. Whether a given model actually
  learns that is what `evaluate.py` measures: in the runs below the 0.5B default model did on this validation set, the 135M
  model mostly did not, and neither result is a guarantee.

In a real system you would combine the two: retrieval supplies the context (with access control applied before the model
sees anything), and a light fine-tune makes the model reliably follow the output contract. That hybrid is what the
video calls the mature enterprise default.

## What is in this folder

| File | What it does |
|---|---|
| `train_lora.py` | LoRA fine-tuning with `transformers` + `peft` + `datasets` (no TRL). Masks the loss to the assistant turn, evaluates on `val.jsonl` every epoch, prints train and validation loss, saves only the adapter. |
| `evaluate.py` | Generates answers for the validation prompts and the general probe, scores base vs tuned side by side, writes `results.json`. Reads the base model from the adapter, so `--model` is optional. |
| `merge_adapter.py` | Optional: merges the adapter into the base model (`merge_and_unload`) and saves a standalone model folder. |
| `common.py` | Shared helpers without torch: JSONL reader, prompt building, label masking, schema checker, JSON parsing, loss-history summary. |
| `lora_finetune.ipynb` | Colab-friendly notebook that mirrors the scripts step by step with short explanations. |
| `data/train.jsonl` | 71 chat examples (20 of them unanswerable, about 28 percent). |
| `data/val.jsonl` | 24 held-out examples (7 unanswerable) that use different passages and questions than training. |
| `data/general_probe.jsonl` | 10 generic prompts with an expected substring (capital cities, small arithmetic, a short definition) for the forgetting check. |
| `tests/` | pytest suite. Needs no torch and no downloads. |
| `requirements.txt` | Python dependencies with minimum versions. |
| `outputs/lora-adapter/` | Created by training: `adapter_config.json`, `adapter_model.safetensors` (the adapter only, not the base model) and `training_summary.json` (hyper-parameters and the per-epoch loss numbers). Git-ignored. |
| `results.json` | Created by `evaluate.py`: metrics for base vs tuned plus every raw model output. Git-ignored. |

All data is fictional and contains no real people, customers or companies. Python 3.10 or newer (only run on 3.12).

## Hardware notes

- **CPU only:** the default 0.5B model trained on CPU in about 10 minutes here (36 optimizer steps, 593 seconds, on an
  8-core / 16-thread AMD Ryzen 7 laptop CPU with 15 GB of RAM and the CPU build of torch); evaluating base, base + prompt and tuned
  took about 7 minutes. Slower machines will take proportionally longer. Fine for learning, tedious for experiments.
- **GPU:** a free Google Colab **T4** should be comfortable for this setup in principle (0.5B model, short sequences,
  fp16 mixed precision, small batches), but no GPU run was done for this starter. If you run out of memory, lower
  `--batch-size` (and raise `--grad-accum`) or `--max-len`.
- The base model is loaded in float32 for stable LoRA training (about 2 GB of weights for 0.5B parameters) and mixed
  precision is used on GPU: bf16 on Ampere or newer, fp16 otherwise. For larger models you may prefer to load in bf16/fp16 to save memory.
- QLoRA = LoRA on a 4-bit-quantized base model to cut memory; this starter implements plain LoRA only.
- The first run downloads the base model (about 1 GB for Qwen2.5-0.5B-Instruct) from the Hugging Face Hub.

## Quick start

Run everything from inside the `finetune-starter` folder.

### Windows PowerShell

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build; for an NVIDIA GPU see pytorch.org
pip install -r requirements.txt
pip install pytest nbformat           # nbformat only for the notebook test (it is skipped without it)

python -m pytest tests                # optional: unit tests, no downloads

# 1. plumbing check with a tiny random model (garbage text, meaningless losses)
python train_lora.py --smoke --model hf-internal-testing/tiny-random-LlamaForCausalLM --out-dir outputs/smoke
python evaluate.py --adapter outputs/smoke --limit 4

# 2. small real model (135M parameters, about 270 MB): the tested small-model commands
python train_lora.py --model HuggingFaceTB/SmolLM2-135M-Instruct --out-dir outputs/smollm2
python evaluate.py --adapter outputs/smollm2 --prompt-baseline

# 3. the default model (0.5B parameters, downloads about 1 GB)
python train_lora.py
python evaluate.py --adapter outputs/lora-adapter --prompt-baseline   # also: base model told the format in words
python merge_adapter.py --adapter outputs/lora-adapter                # optional
```

### macOS / Linux (bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch                     # or the CPU/CUDA build from pytorch.org
pip install -r requirements.txt
pip install pytest nbformat

python -m pytest tests
python train_lora.py --smoke --model hf-internal-testing/tiny-random-LlamaForCausalLM --out-dir outputs/smoke
python evaluate.py --adapter outputs/smoke --limit 4
python train_lora.py --model HuggingFaceTB/SmolLM2-135M-Instruct --out-dir outputs/smollm2
python evaluate.py --adapter outputs/smollm2 --prompt-baseline
python train_lora.py
python evaluate.py --adapter outputs/lora-adapter --prompt-baseline
python merge_adapter.py --adapter outputs/lora-adapter
```

`evaluate.py` reads the base model from the adapter's `adapter_config.json`, so you do not repeat `--model`. If you pass a
`--model` that differs from the recorded one, it prints a warning. If the adapter folder does not exist it stops with
one line: `no adapter at PATH; run train_lora.py first (a smoke run writes to outputs/smoke)`.

`--smoke` uses 8 training and 4 validation examples and stops after 3 optimizer steps. It only proves that the plumbing
works (data loading, masking, LoRA, training loop, saving); a tiny random model produces garbage text, so the script prints
`smoke run: losses are meaningless (plumbing check only)` instead of an interpretation. Running `--smoke` without `--model`
would download the default 1 GB model.

Useful options for `train_lora.py`: `--model`, `--data-dir`, `--out-dir`, `--epochs`, `--lr`, `--rank`, `--alpha`, `--dropout`,
`--batch-size`, `--grad-accum`, `--max-len`, `--max-steps`, `--seed`, `--smoke`. Run `python train_lora.py --help`.
Defaults: 4 epochs, learning rate 2e-4, rank 16, alpha 32, dropout 0.05, batch 4 x accumulation 2 (effective batch 8, 9 optimizer steps per epoch, 36 in total).

### Colab

In Colab use File > Upload notebook and upload `lora_finetune.ipynb`, choose Runtime > Change runtime type > T4 GPU, and run
the cells top to bottom. Colab does not have this folder, so Step 2 asks you to upload `train.jsonl`, `val.jsonl` and
`general_probe.jsonl` from the `data/` folder.

## How to read the results

### Training printout

```
Loss per epoch (train loss = mean during the epoch with dropout on; val loss = measured at epoch end)
epoch | train_loss | val_loss
 1.00 |   ...      |  ...
```

- Both numbers are cross-entropy on the **assistant turn only**: the assistant JSON plus the chat template's end token and
  trailing newline. System and user turns are masked out.
- They should fall together at first.
- **Overfitting signal:** validation loss stops improving or rises while training loss keeps falling. The script prints a
  one-line hint (not for a `--smoke` run). With 71 examples it is easy to memorize the training set; if you see this, use fewer epochs, a lower learning
  rate, a smaller `--rank`, or more varied data.
- Train loss is the running mean during the epoch (dropout on, weights changing) and validation loss is measured at the end,
  so the two are not perfectly comparable. Read the trend, not the gap.
- Low loss alone does not prove the model behaves well. That is what `evaluate.py` is for.
- The raw `{'loss': ...}` dictionaries and progress bars are Hugging Face's own logging; the `[train]` and `[val]` lines are this script's.

### `evaluate.py` and `results.json`

The script prints a table like this, for base and tuned side by side (values omitted here on purpose):

```
metric                                  base   tuned
(a) JSON valid (strict)
    JSON found leniently
(b) schema compliant
(c) escalation ok (unanswerable)
    false escalation (answerable)
(d) general probe pass
    probe outputs that are JSON
```

| Metric | Meaning |
|---|---|
| (a) JSON valid (strict) | The **whole** output parses as one JSON object. Prose around it, or a markdown code fence, counts as invalid. |
| JSON found leniently | An object could be found after ignoring code fences and surrounding prose. Informational. |
| (b) schema compliant | Strictly valid **and** exactly the keys `answer`, `tone` (= `"formal"`), `sources` (list of strings), `needs_escalation` (boolean). |
| (c) escalation ok | On validation examples whose context lacks the answer: share where `needs_escalation` is `true`. Denominator is small (7 in the shipped `val.jsonl`). |
| false escalation | On answerable examples: share where the model escalated anyway. Lower is better. |
| (d) general probe pass | Share of the 10 generic prompts whose answer contains the expected text. A crude check for catastrophic forgetting. |
| probe outputs that are JSON | How often the tuned model answers unrelated questions in the JSON contract. Some leakage of the trained format is expected; it is informational, not a pass/fail. |

What to look for, without expecting particular numbers:

1. **Tuned should show clearly higher (a) and (b) than base.** If it does not, the fine-tune did not take: check the loss curves and hyper-parameters.
2. **(c) above zero without much false escalation.** A model that escalates everything would score 100 percent on (c) and be useless. Learning to abstain can be harder than learning the format: it was for the 135M model in the real run below, and for the earlier 15-percent version of this data.
3. **(d) roughly unchanged.** A large drop is a catastrophic-forgetting warning.
4. `--prompt-baseline` adds a `base+prompt` column: the untouched base model, but with the JSON contract spelled out in the system prompt.
   Try this **before** deciding you need fine-tuning: if prompting alone gets you most of the way, you may not need it.
5. **These metrics do not grade whether the answer is right.** A schema-compliant answer can still be wrong or unfaithful to the
   context. `results.json` contains every raw output (`val_examples`, `probe_examples`); read some of them.

With 24 validation examples (7 unanswerable, 17 answerable) and 10 probe prompts, one example moves a rate by about 4 to 14
percentage points (probe: 10). Decoding is greedy and runs once. Treat the numbers as a smoke signal, and re-measure on your own held-out data.

### One real run, for orientation only

Two real models, both on CPU only (an 8-core AMD Ryzen 7 laptop CPU, 15 GB RAM, torch 2.14.0 CPU build; the machine also has a GPU,
which was not used), seed 42, greedy decoding, one run each, no other setting changed from its default. Counts are out of
24 validation prompts (7 unanswerable, 17 answerable) and 10 probe prompts. Your numbers will differ; one example moves a
rate by 4 to 14 points.

**Default model `Qwen/Qwen2.5-0.5B-Instruct`, shipped defaults** (4 epochs, learning rate 2e-4, rank 16, alpha 32, dropout 0.05,
effective batch 8, 36 optimizer steps; commands: `python train_lora.py`, then `python evaluate.py --adapter outputs/lora-adapter --prompt-baseline`):

| Metric (counts) | base | base + format prompt | tuned |
|---|---|---|---|
| (a) JSON valid (strict) | 0 / 24 | 24 / 24 | 24 / 24 |
| (b) schema compliant | 0 / 24 | 24 / 24 | 24 / 24 |
| (c) escalation ok on unanswerable | 0 / 7 | 7 / 7 | 7 / 7 |
| false escalation on answerable | 0 / 17 | 17 / 17 | 3 / 17 |
| (d) general probe pass | 10 / 10 | not run | 9 / 10 |
| probe outputs that are JSON | 0 / 10 | not run | 1 / 10 |

Training took 593 seconds (about 10 minutes); the three-column evaluation about 7 minutes. Validation loss fell every epoch
(0.360, 0.222, 0.189, 0.185; mean train loss 1.171, 0.238, 0.119, 0.081), so the loss curves showed no sign of overfitting.

What this run does and does not show:

- **The format was learned.** Strictly valid and schema-compliant answers went from 0 of 24 to 24 of 24.
- **Prompting alone also got the format, for this model.** The untouched base model, told the JSON contract in words, produced
  24 of 24 valid objects too. What prompting did not give was usable escalation: it set `needs_escalation` to `true` on all 24
  prompts (17 of them wrongly), and in 10 of 24 outputs its `sources` list named things that are not context ids (for example
  "Context 1"; counted from `results.json`). Run `--prompt-baseline` before deciding that you need fine-tuning for the format alone.
- **Abstention was learned in this run, with caveats.** The tuned model escalated all 7 unanswerable questions, and escalated 3 of
  17 answerable ones by mistake. On an earlier version of this data (64 training examples, about 15 percent unanswerable, 3 unanswerable
  validation examples) the same model and settings had learned the format perfectly but escalated 0 of 3 in a review run, so the
  higher unanswerable share may have made the difference; several things changed at once and there is a single seed, so this
  cannot be attributed. The 7 unanswerable validation examples are written in the same style as the training ones, which is
  easier than real-world questions. The three false escalations read as self-contradictory (the answer restates a fact from the
  context and then says it is not available), and one answerable example (validation 13) says "not available" in words while
  setting the flag to `false`. The metrics only look at the flag: read the raw outputs.
- **No clear forgetting, one possible slip.** The probe went from 10 / 10 to 9 / 10: the tuned model answered "What is 12 plus 15?"
  with "17". One of ten unrelated answers came back in the JSON format. Ten prompts is a weak test.
- **Schema compliance is not correctness.** The correct answers I compared with the gold answers (for example validation examples 0, 1 and 3) were
  near-verbatim restatements of the context sentences, which is what the training targets look like.

**Smaller model `HuggingFaceTB/SmolLM2-135M-Instruct`** (135M parameters, about 270 MB; commands: `python train_lora.py --model HuggingFaceTB/SmolLM2-135M-Instruct --out-dir outputs/smollm2`,
then `python evaluate.py --adapter outputs/smollm2 --prompt-baseline`; the 8-epoch column adds `--epochs 8` and skips `--prompt-baseline`):

| Metric (counts) | base | base + format prompt | tuned, 4 epochs (defaults) | tuned, 8 epochs |
|---|---|---|---|---|
| (a) JSON valid (strict) | 0 / 24 | 0 / 24 | 24 / 24 | 24 / 24 |
| (b) schema compliant | 0 / 24 | 0 / 24 | 24 / 24 | 23 / 24 |
| (c) escalation ok on unanswerable | 0 / 7 | 0 / 7 | 2 / 7 | 1 / 7 |
| false escalation on answerable | 0 / 17 | 0 / 17 | 1 / 17 | 0 / 17 |
| (d) general probe pass | 9 / 10 | not run | 9 / 10 | 9 / 10 |
| probe outputs that are JSON | 0 / 10 | not run | 0 / 10 | 0 / 10 |

Training took about 4 minutes for 4 epochs (232 seconds) and about 8 minutes for 8 epochs (464 seconds); evaluation took about 3 minutes with the prompt baseline and 2 minutes without.
Validation loss fell from 1.46 to 0.60 over 4 epochs and from 1.60 to 0.37 over 8, with no rise.
The small model learned the format (neither the base model nor the prompted base model ever produced a bare JSON object)
but **mostly did not learn to abstain**: 2 of 7 and 1 of 7, and doubling the epochs did not help. Where the flag was set, flag and
answer text did not always agree (for example validation 6 in the 4-epoch run restates the context and still sets the flag).

If escalation is not learned for your model or data, in rough order: add more and more varied unanswerable examples (the
Qwen result above is consistent with, but does not prove, that this helps), try a larger base model, and do not treat
`needs_escalation` as reliable in any case. More epochs alone did not help the 135M model.

## Using your own data

Each line of a JSONL file is `{"messages": [system, user, assistant]}`. The assistant message must be a single JSON string with the
four keys above. This data uses about 28 percent unanswerable examples because an earlier version with about 15 percent did not get
the default model to escalate in one review run (0 of 3, see above); tune the share and check the escalation metric on your own data. Vary topics and phrasing, and hold out
validation examples whose **passages and questions** never appear in training (the tests in `tests/test_data.py` show how to check that).
If you also run a RAG pipeline, keep the subjects and company of your fine-tuning passages different from your retrieval corpus, as this starter does.
Do not put personal, customer or otherwise sensitive data into training files: fine-tuning diffuses it into the weights, where it is hard to remove.
Examples longer than `--max-len` tokens are dropped with a warning (never truncated, because a cut-off JSON target would teach broken JSON).

## Warnings

- **Hyper-parameters are starting points.** The defaults are common LoRA settings for a small model on a small dataset, not tuned values. Change one thing at a time and compare on validation data.
- **Overfitting.** Small datasets are memorized quickly. Watch validation loss.
- **Catastrophic forgetting.** Fine-tuning can degrade abilities you did not train. The 10-prompt probe is crude; test your own generic tasks as well.
- **Evaluate on held-out data.** Never judge a fine-tune on examples it trained on. The validation set here is tiny; build a larger one from your own domain before relying on any conclusion.
- **Behavior is not truth.** A model trained on this data will sound confident and formal even when it is wrong. Keep retrieval, citations and human escalation in the loop.
- **Treat `needs_escalation` as advisory.** Even where escalation was learned above, answerable questions were escalated by mistake
  and one answer said "not available" while the flag said `false`. Verify the flag (for example with a retrieval-score threshold,
  a second check of the answer against the context, or human review) instead of using it as a safety control.
- **Licenses.** Check the license and usage terms of any base model you download.
- Numbers you may remember from the video (cost and latency ranges) are illustrative ranges, not benchmarks of this repository. Measure on your own stack.

## What was and was not tested

Tested when this folder was revised (Windows 11, Python 3.12, CPU only, torch 2.14.0, transformers 5.17.0, peft 0.21.0, datasets 5.0.1, accelerate 1.15.0):

- Unit tests in `tests/`: 79 passed with the packages above plus pytest and nbformat; 78 passed and 1 skipped (the notebook
  validation) in an environment with only pytest and no torch, transformers or nbformat. They cover data validity, train/val
  separation (prompts, questions and passages), the check against the RAG documents, the masking helper with fake tokenizers,
  the schema checker, scoring, base-model resolution, start-up messages, and notebook well-formedness, including that the
  notebook's probe check agrees with `common.py`. `pyflakes` is clean and every file parses as Python 3.10 syntax (not run on 3.10).
- The masking helper against the real Qwen2.5 tokenizer and chat template (tokenizer files only): the trained span is the
  assistant JSON plus the `<|im_end|>` token and the trailing newline. The longest example is 214 tokens, so nothing is dropped at the default `--max-len 512`.
- Smoke runs of `train_lora.py --smoke`, `evaluate.py --adapter` (base model read from the adapter) and `merge_adapter.py` with the
  tiny random model `hf-internal-testing/tiny-random-LlamaForCausalLM`, and a smoke train plus evaluate with `sshleifer/tiny-gpt2`,
  which has no chat template and so exercises the plain fallback format. Tiny random models produce garbage text; these runs only prove that the code path executes.
- Complete train-and-evaluate runs on `Qwen/Qwen2.5-0.5B-Instruct` (defaults) and `HuggingFaceTB/SmolLM2-135M-Instruct`
  (4 and 8 epochs), CPU only: the numbers above.
- The notebook's code cells (all but the `%pip` install cell) executed locally, outside Colab, against the tiny Llama model.

NOT tested: any GPU or Colab run (so the fp16/bf16 branches and `.to(cuda)` paths never executed); other seeds (every number
above is one run with seed 42); transformers/peft versions other than those listed; Python 3.10 and 3.11; macOS or Linux (the code uses
`pathlib` and ASCII console output and should work, but it was only run on Windows); the missing-dependency message with a
genuinely uninstalled package (it is unit-tested by blocking the imports); a differing `--model` with a working adapter (the warning was seen on the command line with a mismatched tiny model,
where loading then failed as the warning says; the resolution logic is unit-tested).

## Troubleshooting

- **`no adapter at PATH; run train_lora.py first ...`.** `evaluate.py` and `merge_adapter.py` need a folder that contains `adapter_config.json`. Train first (a smoke run writes to `outputs/smoke`, a normal run to `outputs/lora-adapter`) or fix the path.
- **`missing Python package '...'; install the dependencies first: pip install -r requirements.txt`.** Activate the virtual environment where you installed the requirements, and install torch as shown in the quick start.
- **`pip install` fails with "No such file or directory" on a very long path (Windows).** Windows limits paths to 260 characters
  unless long paths are enabled. Create the virtual environment in a folder with a short path (directly under the drive root, for example), or enable Windows long-path support.
- **Out of memory.** Lower `--batch-size` and `--max-len`, and raise `--grad-accum` to keep the effective batch size.
- **Very slow on CPU.** Use Colab, or try `--max-steps 20` to see the loop run before committing to a full run.
- **Console shows odd characters.** Output from this repository is ASCII only; third-party progress bars may differ.

## License

MIT. See the `LICENSE` file in the repository root.
