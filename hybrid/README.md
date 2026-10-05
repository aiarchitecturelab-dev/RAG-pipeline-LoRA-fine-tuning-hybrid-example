# Hybrid example: RAG for the facts, LoRA for the behavior

**This is what Part 2 of the video series walks through**, end to end: training the LoRA adapter live (real loss curve, real
timing), then running all three demo questions below with the freshly trained adapter and reading the real validation output.

Part 1 calls this the mature enterprise default: **retrieval supplies the facts** (fresh, cited, permission-filtered), and a
**light LoRA fine-tune supplies the behavior** (formal tone, one strict JSON object, an escalation flag).
`hybrid_example.py` is a short, commented script that connects the two halves of this repository:

```text
question + role
   -> rag-starter: permission filter, search, re-rank          (the caller's allowed chunks only)
   -> numbered CONTEXT block, same format as finetune-starter/data/train.jsonl
   -> base model + LoRA adapter (transformers + peft)          (greedy decoding)
   -> one JSON object: answer, tone, sources, needs_escalation
   -> validation: valid JSON? required keys? cited ids retrieved? answer consistent with its own flag?
```

Diagram and stage-by-stage explanation: [`../docs/architecture.md`](../docs/architecture.md#3-the-hybrid-pipeline-hybrid).
Everything is fictional. The script reuses code from both starters, so keep the repository layout as it is:
`hybrid/`, `rag-starter/` and `finetune-starter/` side by side.

One thing to know before you read the outputs: the retrieved documents are about **Acme Corp**, but the fine-tuning data (and
therefore the adapter's system message) is about a different fictional company, **Globex Corp**, on purpose (see
[`../finetune-starter/README.md`](../finetune-starter/README.md)). `hybrid_example.py` uses the exact system message the adapter was
trained with, so the prompt below says "Globex Corp" above Acme text. That is a demo shortcut; in your own project, train and run with your own system message.

## Try it without a model: `--dry-run`

`--dry-run` prints the retrieved sources and the **exact prompt** the model would receive, then stops. It needs only `numpy`:
no torch, no transformers, no peft, no model download, no API key, and no network after `pip install`. Run these from the **repository root**
(if you already did the "Start here" steps in the top-level README, you have this environment: skip everything above the blank line).

**Windows PowerShell**

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

pip install -r rag-starter/requirements.txt
python rag-starter/ingest.py --docs rag-starter/data/docs --index rag-starter/index
python hybrid/hybrid_example.py "What's our parental leave policy?" --role employee --dry-run
```

**macOS / Linux (bash)**

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r rag-starter/requirements.txt
python rag-starter/ingest.py --docs rag-starter/data/docs --index rag-starter/index
python hybrid/hybrid_example.py "What's our parental leave policy?" --role employee --dry-run
```

Real output of the last command (the three `CONTEXT` lines are shortened here with `...`; the real output prints each chunk in full):

```text
Question : What's our parental leave policy?
Role     : employee (may read access levels: all)
Retrieval: 7 candidates -> kept 3 after re-ranking

========================================================================
RETRIEVED SOURCES (permission filter, similarity search, re-ranking)
========================================================================
[SRC-01] score=0.490  [Source: hr-policy-2026.md, Parental leave]
[SRC-02] score=0.421  [Source: hr-policy-2026.md, Acme Corp HR Policy 2026]
[SRC-03] score=0.275  [Source: hr-policy-2026.md, Sick leave]

========================================================================
SYSTEM PROMPT (exact text)
========================================================================
You are the Globex Corp internal policy assistant.

========================================================================
USER PROMPT (exact text)
========================================================================
CONTEXT:
[SRC-01] Parental leave: Acme Corp provides 12 weeks of paid parental leave to every eligible employee ...
[SRC-02] Acme Corp HR Policy 2026: This policy describes working conditions and leave entitlements ...
[SRC-03] Sick leave: Employees receive up to 10 days of paid sick leave per year. ...

QUESTION: What's our parental leave policy?

[dry run] No model was loaded and nothing was generated.
```

Results 2 and 3 are weak matches (the default embedder matches words, not meaning), which is realistic; a fine-tuned model has to cope with
irrelevant context, and the small demo model below does not always manage.

Try the permission filter: ask `"What is the CEO's base salary?"` with `--role employee` and then `--role executive`.
The employee's prompt contains no text from `exec-compensation-2026.md`, because those chunks are filtered out before the prompt exists.

## Run it with a model

You need (1) an environment with torch, transformers and peft, and (2) a LoRA adapter you trained with
[`finetune-starter`](../finetune-starter/README.md). The adapter only fits the base model it was trained on; the script reads that
model's name from the adapter's `adapter_config.json`, so `--model` is usually not needed.

The recipe below is the one that was run for this page: a **small 135M-parameter model** (about 270 MB to download; 8 epochs took about
8 minutes on an 8-core laptop CPU). The adapter behind the outputs below was trained with exactly these options, saved to a different folder and then copied to the
`--adapter` path shown. Continue in the same virtual environment, from the repository root.

**Windows PowerShell**

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU build; for an NVIDIA GPU see pytorch.org
pip install -r finetune-starter/requirements.txt

python finetune-starter/train_lora.py --model HuggingFaceTB/SmolLM2-135M-Instruct --epochs 8 --out-dir finetune-starter/outputs/smollm2-8ep
python hybrid/hybrid_example.py "What's our parental leave policy?" --role employee --adapter finetune-starter/outputs/smollm2-8ep
```

**macOS / Linux (bash)**

```bash
pip install torch                         # or the CPU/CUDA build from pytorch.org
pip install -r finetune-starter/requirements.txt

python finetune-starter/train_lora.py --model HuggingFaceTB/SmolLM2-135M-Instruct --epochs 8 --out-dir finetune-starter/outputs/smollm2-8ep
python hybrid/hybrid_example.py "What's our parental leave policy?" --role employee --adapter finetune-starter/outputs/smollm2-8ep
```

To use the default 0.5B model instead (downloads about 1 GB; about 10 minutes of training on the same CPU), run `python finetune-starter/train_lora.py`
with no options. Then `--adapter` can be left out, because the script looks in `finetune-starter/outputs/lora-adapter`, where `train_lora.py` saves by default.
Both variants were run for this page (results below). To use an adapter and model from somewhere else (paths and model name are arguments):

```bash
python hybrid/hybrid_example.py "What's our parental leave policy?" --role employee --adapter path/to/my-adapter --model Qwen/Qwen2.5-0.5B-Instruct
```

### Real output, and how to read it

The three runs below used the 8-epoch 135M adapter from the recipe above (trained on the current `finetune-starter` data, seed 42,
greedy decoding; repeating a run gave identical output). The Windows console prints the adapter path with backslashes. **A 135M model is a
demonstration that the code path works, not a claim about answer quality**: expect poor answers from it, and expect a real model and your own
training data to behave differently. Nothing here is a benchmark.

**1. The video's question, as an employee.** The output is well-formed and passes validation, but it is not a good answer:

```text
Question : What's our parental leave policy?
Role     : employee (may read access levels: all)
Retrieval: 7 candidates -> kept 3 after re-ranking

========================================================================
RETRIEVED SOURCES (permission filter, similarity search, re-ranking)
========================================================================
[SRC-01] score=0.490  [Source: hr-policy-2026.md, Parental leave]
[SRC-02] score=0.421  [Source: hr-policy-2026.md, Acme Corp HR Policy 2026]
[SRC-03] score=0.275  [Source: hr-policy-2026.md, Sick leave]

Generating with HuggingFaceTB/SmolLM2-135M-Instruct + adapter finetune-starter\outputs\smollm2-8ep ...
========================================================================
MODEL OUTPUT (raw)
========================================================================
{"answer": "The provided policy describes working conditions and leave entitlements for all employees of Acme Corp. It applies to full-time and part-time staff in every office.", "tone": "formal", "sources": ["SRC-01"], "needs_escalation": false}

========================================================================
VALIDATION: PASS
========================================================================
Answer          : The provided policy describes working conditions and leave entitlements for all employees of Acme Corp. It applies to full-time and part-time staff in every office.
needs_escalation: False
  SRC-01 = [Source: hr-policy-2026.md, Parental leave]
```

The answer restates the introduction chunk (`SRC-02`), not the 12-week rule in `SRC-01`, and yet it cites `SRC-01`. The validator cannot see that:
the JSON is valid and `SRC-01` was retrieved. This is the gap between "well formed" and "correct" (see "What the validation checks, and what it does not" below).

**2. "What is the CEO's base salary?" as an employee.** Retrieval returns only unrelated HR chunks; no executive text reaches the prompt. The
model answers in a self-contradicting way, and validation now catches it (exit code 2):

```text
Question : What is the CEO's base salary?
Role     : employee (may read access levels: all)
Retrieval: 2 candidates -> kept 2 after re-ranking

========================================================================
RETRIEVED SOURCES (permission filter, similarity search, re-ranking)
========================================================================
[SRC-01] score=0.307  [Source: hr-policy-2026.md, Parental leave]
[SRC-02] score=0.175  [Source: hr-policy-2026.md, Performance reviews]

Generating with HuggingFaceTB/SmolLM2-135M-Instruct + adapter finetune-starter\outputs\smollm2-8ep ...
========================================================================
MODEL OUTPUT (raw)
========================================================================
{"answer": "The requested information is not available in the provided context. The provided policy states that parental leave is provided for all parents, regardless of gender or of whether they are the birth parent.", "tone": "formal", "sources": ["SRC-01"], "needs_escalation": false}

========================================================================
VALIDATION: FAIL
========================================================================
  - inconsistent: says the information is unavailable but needs_escalation is false
```

The permission filter did its job. The model's answer says the information is missing, but it also cites a source and leaves `needs_escalation` at `false`.
Checks 1 to 4 (listed below) would all have passed this output; only the consistency check (check 5) rejects it. A weak model can also produce a confident, well-formed answer
that is simply wrong and passes every check, so treat validation as a floor, not a guarantee.

**3. The same question as an executive.** Now the salary chunk is retrieved (`SRC-01`, "Base salary bands"), and it does contain the figure. The model still says the information
is not available, and escalates. This passes validation, because the text and the flag agree:

```text
Question : What is the CEO's base salary?
Role     : executive (may read access levels: all, executive, managers)
Retrieval: 7 candidates -> kept 3 after re-ranking

========================================================================
RETRIEVED SOURCES (permission filter, similarity search, re-ranking)
========================================================================
[SRC-01] score=0.553  [Source: exec-compensation-2026.md, Base salary bands]
[SRC-02] score=0.382  [Source: exec-compensation-2026.md, Severance and change of control]
[SRC-03] score=0.337  [Source: exec-compensation-2026.md, Executive benefits]

Generating with HuggingFaceTB/SmolLM2-135M-Instruct + adapter finetune-starter\outputs\smollm2-8ep ...
========================================================================
MODEL OUTPUT (raw)
========================================================================
{"answer": "The requested information is not available in the provided context. The provided policy states that the CEO's base salary is approved annually, but it does not mention any annual base salary bands. The provided policy states that the Chief Executive Officer receives a supplemental pension contribution of 12 percent of base salary and an annual health assessment.", "tone": "formal", "sources": [], "needs_escalation": true}

========================================================================
VALIDATION: PASS
========================================================================
Answer          : The requested information is not available in the provided context. The provided policy states that the CEO's base salary is approved annually, but it does not mention any annual base salary bands. The provided policy states that the Chief Executive Officer receives a supplemental pension contribution of 12 percent of base salary and an annual health assessment.
needs_escalation: True
```

This is a false escalation: safe in the sense that a human would take over, but wrong, and no validator here can tell that the answer was in the context.
`finetune-starter` measured that this model learned the JSON format but not reliable abstention, and its errors there mostly ran the other way (it rarely escalated on its
short, clean validation contexts). Here, with longer retrieved chunks that look unlike the training data, it escalated wrongly: the "known gap" in the design notes.

### The same three questions with the default 0.5B model

For comparison, the adapter from `python finetune-starter/train_lora.py` (the default `Qwen/Qwen2.5-0.5B-Instruct`, 4 epochs, seed 42), used without `--adapter`.
Retrieval is identical to the runs above, so only the model output and the validation result are shown, exactly as printed. One run per question;
three questions are not a benchmark.

**1. Parental leave, as an employee.** Passes. The answer restates the first three sentences of `SRC-01` and cites it (it leaves out the eligibility and notice rules):

```text
{"answer": "The provided policy states that Acme Corp provides 12 weeks of paid parental leave to every eligible employee who becomes a parent through birth, adoption, or foster placement. The entitlement is the same for all parents, regardless of gender or of whether they are the birth parent. Leave is paid at 100 percent of base salary.", "tone": "formal", "sources": ["SRC-01"], "needs_escalation": false}
VALIDATION: PASS
```

**2. CEO's base salary, as an employee.** Passes, and this is the behavior you want: no figure, no citation, an escalation. Nothing about executive pay was in the prompt, so nothing could leak:

```text
{"answer": "The requested information is not available in the provided context. The provided policy describes paid parental leave and performance reviews but does not mention the CEO's base salary. This question should be escalated to the relevant policy owner for an authoritative answer.", "tone": "formal", "sources": [], "needs_escalation": true}
VALIDATION: PASS
```

**3. CEO's base salary, as an executive.** Fails validation (exit code 2). The number 620,000 is the one in the executive document (the model added a dollar sign that the document does not have),
but the same answer also says the information is not available while the flag says `false`:

```text
{"answer": "According to the provided policy, the Chief Executive Officer's base salary is $620,000. The requested information is not available in the provided context.", "tone": "formal", "sources": ["SRC-01"], "needs_escalation": false}
VALIDATION: FAIL
  - inconsistent: says the information is unavailable but needs_escalation is false
```

The check flagged the contradiction, not the number, and nothing here would catch an added dollar sign. It fails safe: a real system would retry or escalate, and here it costs a false alarm on an answer that was mostly right.
So the default model behaved much better than the 135M one on these three questions, and validation still had work to do.

### Options

| Option | Default | Meaning |
|---|---|---|
| `question` | (required) | The question, in quotes. |
| `--role` | `employee` | `employee`, `manager` or `executive`. Decides which chunks retrieval may return. |
| `--index` | `rag-starter/index` | Index built by `rag-starter/ingest.py`. |
| `--top-n` | `3` | Chunks put into the prompt. Small on purpose: the adapter was trained on short contexts. |
| `--adapter` | `finetune-starter/outputs/lora-adapter` | LoRA adapter folder. |
| `--model` | the base model recorded in the adapter | Base model id or local folder. A warning is printed if it differs from the recorded one. |
| `--max-new-tokens` | `220` | Generation limit. |
| `--dry-run` | off | Print retrieval and the exact prompt, then stop. No model is loaded. |

Exit codes: `0` fine, `1` setup problem (no index, no adapter, torch missing, ...), `2` the model's output failed validation.

## What the validation checks, and what it does not

`validate_output()` requires that

1. the **whole** output is one JSON object (a code fence or extra prose fails);
2. it has exactly the keys `answer`, `tone` (`"formal"`), `sources` (a list of strings) and `needs_escalation` (a boolean);
3. every id in `sources` (`SRC-01`, ...) is one that was actually retrieved and shown to the model;
4. an answer that is not an escalation cites at least one source;
5. an answer that is not an escalation does not say the information is unavailable. This is a keyword screen: if `needs_escalation` is `false` and
   the answer contains a phrase such as "not available in the provided context", "does not mention", "does not contain", "cannot find" or "no information"
   (case and spacing ignored; the full list is `ABSTENTION_PHRASES` in the script), validation fails with
   `inconsistent: says the information is unavailable but needs_escalation is false`.

That proves the output is **well formed**, **points at retrieved text** and does not contradict its own flag in the obvious ways. It does **not** prove the answer
is true, or that the cited passage supports it, as run 1 and run 3 above show. Check 5 in particular:

- catches only the listed phrases, not every way of saying "I do not know";
- can, rarely, flag a correct answer that happens to contain one of the phrases. That fails safe (the answer is retried or escalated), but it is a false alarm;
- does not look at the opposite mistake (an escalation whose text answers the question, or a false escalation as in run 3).

Two things follow from the runs above:

- The permission filter is the protection that matters: no executive text reached the employee's prompt, so nothing restricted could leak. That is why access control belongs in retrieval.
- Well-formed output is not correct output. In production add grounding checks (for example: numbers in the answer must appear in the cited chunk; an entailment or judge model; human review for high-stakes answers),
  keep the `needs_escalation` path, and measure the whole pipeline on your own questions.

## Design notes

- **Same prompt as training.** `build_messages()` writes each chunk as `[SRC-01] Section: text` on one line and the user message as `CONTEXT:` + lines + blank line + `QUESTION: ...`, with the same system message as `finetune-starter/data/*.jsonl` (a test compares the two). `common.encode_prompt()` (used by training and `evaluate.py`) renders it, so the model sees the format it was tuned on.
- **Ids are mapped back.** The model cites ids, not file names; a dictionary turns each id into `[Source: file, section]` (or `p.12` for PDF pages) for display.
- **Lazy imports.** `torch`, `transformers` and `peft` are imported inside `generate()` only. A test checks that a dry run does not load them.
- **Known gap.** The shipped training data has short, clean passages; retrieved chunks are longer and noisier. Add training examples that look like your real retrieved context (including irrelevant chunks and unanswerable questions) before relying on the adapter.
- **Not for production as is.** The model is reloaded on every run, there is no retry on invalid JSON, no logging or audit trail, and no batching.

## Tests

```powershell
pip install -r rag-starter/requirements-dev.txt
python -m pytest hybrid
```

40 tests, no torch needed: prompt shape versus the training data, id-to-citation mapping, an employee prompt never containing
executive-only text (and an executive prompt containing it, so the check is not vacuous), every validation rule including the consistency check
(each phrase, case and contraction handling, and normal answers that must not be flagged), error messages,
and that a dry run loads no heavy library. The generation step is replaced by a stub in these tests, so they say nothing about
what a real model produces.

## What was and was not tested

Tested (Windows 11, Python 3.12, CPU only; torch 2.14.0, transformers 5.17.0, peft 0.21.0 for the live runs):

- `--dry-run` with the PowerShell commands above (the bash versions were not run), in a fresh virtual environment that had only the packages from `rag-starter/requirements.txt` (no torch, transformers or peft).
- The 40 tests above.
- The live path with two real, small models, three questions each, one run per question (not a benchmark):
  - `HuggingFaceTB/SmolLM2-135M-Instruct` plus an adapter trained by `train_lora.py --epochs 8` on the shipped data. Repeating its three runs gave identical output. One run passed validation with a poor answer, one failed validation, one passed with a wrong escalation.
  - The default `Qwen/Qwen2.5-0.5B-Instruct` plus the adapter from `train_lora.py` with no options, used without `--adapter`. One good answer, one correct escalation, one self-contradiction that failed validation.
- The failure path, with an adapter trained on a tiny random model (`--smoke`): garbage output was rejected (exit code 2).

**Not** tested: any GPU, macOS or Linux, other transformers or peft versions, other seeds, more than these three questions, and any model that was not trained on this repository's data.
