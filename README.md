# RAG vs Fine-Tuning starter template

Watch the videos: Part 1, "RAG vs Fine-Tuning for Enterprise LLMs" — coming soon. Part 2, "We Built the RAG + Fine-Tune
Hybrid" — coming soon.

Starter code and a decision guide for the video series **"RAG vs Fine-Tuning for Enterprise LLMs"**.

**Part 2 update:** the `hybrid/` pipeline below is no longer just described in this README — it's the subject of its own
video, which trains a real LoRA adapter on camera (loss, timing and adapter size are all from a real run, not slides) and
runs all three `hybrid_example.py` demo questions shown in `hybrid/README.md`. See [`hybrid/README.md`](hybrid/README.md)
for the walkthrough.

The video argues that a base LLM's knowledge is frozen at training time. **RAG** (retrieval-augmented generation) gives it a
live lookup: embed the question, search pre-embedded document chunks, put the best chunks into the prompt, and answer with
citations, with permissions enforced at retrieval time. **Fine-tuning** (LoRA) changes the model's weights, which makes it a
tool for *behavior* (tone, style, strict JSON or tool-call formats), not for facts. This repository lets you run both halves on
a small fictional example: an internal assistant at **Acme Corp** that answers *"What's our parental leave policy?"* with a
citation, and whose retrieval must never let an ordinary employee (for example an intern) reach executive-compensation text.
A two-page decision matrix helps you choose, and a short hybrid example shows the pattern the video calls the mature
enterprise default: RAG for the facts plus a light LoRA fine-tune for the behavior.

Everything is fictional (Acme Corp, Globex Corp, their people, policies and figures) and runs on a laptop. The default RAG demo needs
no API key, and no network after `pip install`. Numbers quoted in the video and in the PDF are **illustrative ranges, not
benchmarks**: measure on your own stack.

## Start here (about 1 minute, no API key, no GPU)

**1. Get the code.** On the repository's page choose Code, then Download ZIP, and unzip it (or run `git clone` with the repository's
address). Put it in a folder with a short path, such as `C:\src` on Windows. Then open PowerShell in the folder that contains this
README: in File Explorer, click the address bar, type `powershell` and press Enter.

**2. Run these commands** (Python 3.10 or newer; macOS and Linux users, open the bash version right below):

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

pip install -r rag-starter/requirements.txt
python rag-starter/ingest.py --docs rag-starter/data/docs --index rag-starter/index
python rag-starter/query.py "What's our parental leave policy?" --role employee --dry-run --index rag-starter/index
```

The `Set-ExecutionPolicy` line affects only this PowerShell window; it lets PowerShell run the activation script. The last three lines are
the work: install, build the index, ask a question. `--dry-run` sends nothing anywhere.

<details>
<summary>macOS / Linux (bash)</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r rag-starter/requirements.txt
python rag-starter/ingest.py --docs rag-starter/data/docs --index rag-starter/index
python rag-starter/query.py "What's our parental leave policy?" --role employee --dry-run --index rag-starter/index
```

</details>

**3. What you should see.** After a short header, the last command lists five ranked sources (real output), and then prints the exact prompt it would send to a model:

```text
[1] score=0.490 (vector 0.226)  hr-policy-2026.md | Parental leave | access=all | updated=2026-01-15
[2] score=0.421 (vector 0.219)  hr-policy-2026.md | Acme Corp HR Policy 2026 | access=all | updated=2026-01-15
[3] score=0.275 (vector 0.188)  hr-policy-2026.md | Sick leave | access=all | updated=2026-01-15
[4] score=0.272 (vector 0.179)  it-security-policy.md | Acme Corp IT Security Policy | access=all | updated=2026-02-03
[5] score=0.263 (vector 0.157)  expense-policy.md | Acme Corp Expense Policy | access=all | updated=2026-03-01
```

The top result is the **Parental leave** section, which is the right answer. Results 2 to 5 are weak matches, because the default embedder matches
words, not meaning (every document's introduction contains the word "policy"). That is expected, and it is why the prompt tells a model to ignore
irrelevant sources and to say it does not know when the sources are not enough.

Two Windows notes:

- If `python` is not found on Windows, try `py -3.12` (for example `py -3.12 -m venv .venv`).
- If `pip install` fails with `No such file or directory` on a long path, move or clone the folder to a short path such as `C:\src`, or enable Windows long paths.

## What needs what

| Part | API key | GPU | Network |
|---|---|---|---|
| RAG, dry run (Start here above) | No | No | Only for `pip install` |
| RAG, live answers | Yes (`ANTHROPIC_API_KEY`) | No | Yes (calls the Anthropic API) |
| Fine-tuning (LoRA) | No | Optional (CPU works, slowly) | Downloads a base model from the Hugging Face Hub (about 270 MB or 1 GB) |
| Hybrid, dry run | None | None | None after `pip install` |
| Hybrid, live (needs a trained adapter) | No | Optional | Downloads the base model, as for fine-tuning |

## What is in this repository

```text
RAG-pipeline-LoRA-fine-tuning-hybrid-example/
  README.md                 this file
  LICENSE                   MIT
  .gitignore
  docs/
    architecture.md         Mermaid diagrams, a glossary and a stage-by-stage explanation of all three pipelines
    decision-matrix.pdf     two-page cheat sheet: 10-criterion matrix, decision flow, failure modes, checklist
  rag-starter/              the RAG half (Python, numpy only for the default demo)
    README.md               full guide: pipeline map, access-control demo, evaluation, plugging in real components
    ingest.py               documents -> chunks -> vectors -> index
    query.py                question -> permission filter -> re-rank -> prompt -> answer (or --dry-run)
    rag/                    one module per pipeline step
    data/docs/              four fictional Acme Corp documents (one is executive-only)
    data/eval_questions.jsonl
    evals/eval_retrieval.py hit@k evaluation and chunk-size sweep
    tests/                  offline pytest suite, including access-control tests
    requirements.txt  requirements-dev.txt  .env.example  pytest.ini
  finetune-starter/         the fine-tuning half (LoRA with transformers + peft)
    README.md               full guide: hardware notes, real results and how to read them, warnings
    train_lora.py           LoRA training, loss masked to the assistant reply, saves the adapter only
    evaluate.py             base vs tuned: JSON validity, schema, escalation, forgetting probe
    merge_adapter.py        optional: merge the adapter into a standalone model
    common.py               shared helpers (no torch needed)
    lora_finetune.ipynb     Colab-friendly notebook that mirrors the scripts
    data/                   train.jsonl (71 examples), val.jsonl (24), general_probe.jsonl (10); fictional "Globex Corp" text
    tests/                  pytest suite, no torch needed
    requirements.txt
  hybrid/                   RAG retrieval + the LoRA adapter, with output validation
    README.md
    hybrid_example.py       --dry-run works with numpy only
    test_hybrid_example.py
```

## More RAG: the permission filter and live answers

The commands in this section run from the repository root, like the ones above, with the environment from "Start here" still active.

Try the permission filter. The employee never retrieves executive-compensation text; the executive does:

```powershell
python rag-starter/query.py "What is the CEO's base salary?" --role employee  --dry-run --explain-access --index rag-starter/index
python rag-starter/query.py "What is the CEO's base salary?" --role executive --dry-run --explain-access --index rag-starter/index
```

For the employee, the sources are two unrelated HR chunks and the debug line reports that the filter removed 9 chunks. For the executive, the top
five sources all come from `exec-compensation-2026.md` (the first is its "Base salary bands" section). `--explain-access` is a debugging aid: never show
it to end users, because it reveals that restricted documents exist.

For a real answer with a citation check, set the `ANTHROPIC_API_KEY` environment variable and drop `--dry-run`. Live calls were **not** run while building
this repository (no API key was available). `query.py` and `ingest.py` read a `.env` file from the current folder, so if you use one, work inside
`rag-starter` as its README describes. Details, settings, the retrieval evaluation and how to plug in real embeddings: [`rag-starter/README.md`](rag-starter/README.md).

## Quick start: fine-tuning (LoRA)

Fine-tuning teaches a model *how to answer*, not what is true. Here it is trained to reply in a formal tone as one strict JSON object, and (in
the "unanswerable" training examples) to say the context lacks the answer and set an escalation flag. Whether a given model actually learns that last
part varies a lot; the measured results are below. Read [`finetune-starter/README.md`](finetune-starter/README.md) before trusting any number it prints.

Needs Python 3.10 or newer and PyTorch, and downloads a base model. Commands run from the repository root with the same environment as above. On a CPU the
small model is the practical choice; a free Colab T4 GPU is comfortable in principle (see `lora_finetune.ipynb`), but no GPU or Colab run was made for this repository.

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build; for an NVIDIA GPU see pytorch.org
pip install -r finetune-starter/requirements.txt

# A. The tested small-model commands: 135M parameters, about 270 MB download, about 4 minutes of training on an 8-core laptop CPU
python finetune-starter/train_lora.py --model HuggingFaceTB/SmolLM2-135M-Instruct --out-dir finetune-starter/outputs/smollm2
python finetune-starter/evaluate.py --adapter finetune-starter/outputs/smollm2 --prompt-baseline

# B. The default model: 0.5B parameters, about 1 GB download, about 10 minutes of training and 7 minutes of evaluation on the same CPU
python finetune-starter/train_lora.py
python finetune-starter/evaluate.py --adapter finetune-starter/outputs/lora-adapter --prompt-baseline
```

<details>
<summary>macOS / Linux (bash)</summary>

```bash
pip install torch                     # or the CPU/CUDA build from pytorch.org
pip install -r finetune-starter/requirements.txt

python finetune-starter/train_lora.py --model HuggingFaceTB/SmolLM2-135M-Instruct --out-dir finetune-starter/outputs/smollm2
python finetune-starter/evaluate.py --adapter finetune-starter/outputs/smollm2 --prompt-baseline

python finetune-starter/train_lora.py
python finetune-starter/evaluate.py --adapter finetune-starter/outputs/lora-adapter --prompt-baseline
```

</details>

`evaluate.py` reads the base model from the adapter, so you do not repeat `--model`. For a seconds-long plumbing check with a tiny random model, an optional
adapter merge and the notebook, see the fine-tune README. QLoRA = LoRA on a 4-bit-quantized base model to cut memory; this starter implements plain LoRA only.

### What these runs measured

One run each, seed 42, CPU only, defaults except where stated, greedy decoding, on 24 validation prompts (7 whose context lacks the answer, 17 answerable). One example
moves a rate by 4 to 14 points, so read these as a smoke signal, not evidence:

| Result (counts) | SmolLM2-135M (4 epochs) | Qwen2.5-0.5B, the default |
|---|---|---|
| Strictly valid JSON: base model, then tuned | 0 of 24, then 24 of 24 | 0 of 24, then 24 of 24 |
| Escalated the 7 unanswerable questions (tuned) | 2 of 7 | 7 of 7 |
| Escalated an answerable question by mistake (tuned) | 1 of 17 | 3 of 17 |

What that does and does not show:

- **Both models learned the format.** This is the part fine-tuning is good at.
- **Abstention was learned by the larger model, and mostly not by the smaller one.** So "the fine-tuned model says so when the context lacks the answer" is true here for
  Qwen2.5-0.5B on this small, same-style validation set, with 3 wrong escalations, and mostly not true for the 135M model. Treat `needs_escalation` as advisory, not as a safety control.
- **Prompting alone also got the format from the default model** (24 of 24 valid objects when the JSON contract was spelled out in the prompt), though not usable escalation. That is why
  `--prompt-baseline` exists: try it before deciding you need fine-tuning.
- **Schema-compliant is not correct.** Read the raw outputs in `results.json`.

Full tables, the loss curves, the 8-epoch run and every caveat: [`finetune-starter/README.md`](finetune-starter/README.md#one-real-run-for-orientation-only).
The training data is about "Globex Corp" and covers different topics than the RAG documents on purpose, so the adapter cannot memorize a fact that retrieval supplies.

## Which one do I need?

The full answer is the two-page cheat sheet, [`docs/decision-matrix.pdf`](docs/decision-matrix.pdf): ten criteria compared across
RAG, LoRA fine-tuning and the hybrid, a decision flow, the failure modes of each approach, and a six-step checklist to run before you build.
The short version:

| If you need... | Reach for | Why |
|---|---|---|
| Answers that follow documents that change (policies, product docs) | RAG | Update a document, re-ingest it, and answers change. No retraining. |
| Citations and an audit trail | RAG | Every answer can name its source chunk and page. A fine-tuned model has no source to point to. |
| Different users to see different documents | RAG with a permission filter | Filter chunks by permission at retrieval time, before the LLM sees them. Model weights are shared by every user. |
| Sensitive data you may need to remove | RAG | Delete the document and re-index, and it is no longer retrieved (also purge logs, caches and backups). Data trained into weights is hard to remove. |
| A consistent tone, style or strict output format (JSON, tool calls) | A light LoRA fine-tune | Fine-tuning is for behavior, not facts. |
| Fresh cited facts **and** a fixed format | Hybrid | RAG for facts, LoRA for behavior. |

The decision flow from the PDF, in order: (A) can better prompts, few-shot examples or structured outputs already give you the
behavior? Then skip fine-tuning for now, but still answer B and C. (B) Does the knowledge change often, or must answers be cited?
Use RAG. (C) Do you need per-user permissions? Enforce them in retrieval, never in the model. (D) Do you need a tone, format or
tool-calling behavior that prompting cannot deliver reliably? Add a light LoRA fine-tune, for behavior only. If B or C *and* D apply, combine them (the hybrid).
Before building anything: write 20-50 real questions with their expected sources, start with RAG plus good prompts, and measure retrieval before touching the model.

## The hybrid pattern

```text
question + role -> permission filter -> search -> re-rank -> numbered context
                -> base model + LoRA adapter -> one JSON object -> validate -> answer with citations
```

RAG supplies the facts, so they stay fresh and cited, and permissions are enforced before the model sees anything. The adapter
supplies a consistent tone and output contract. Change a policy document and re-ingest: the answers change without retraining.
[`hybrid/hybrid_example.py`](hybrid/hybrid_example.py) is a short commented example. Its `--dry-run` shows the exact prompt using numpy only:

```powershell
python hybrid/hybrid_example.py "What's our parental leave policy?" --role employee --dry-run
```

The live path (a trained adapter plus torch) is described in [`hybrid/README.md`](hybrid/README.md), with real outputs for three questions from two small models. These are a code-path demo, **not a
quality claim**. With the 135M-parameter model, one answer was well formed but poor, one failed the output check, and one was a wrong escalation. With the default 0.5B model, one answer was correct and cited,
one was a correct escalation, and one contradicted itself and was rejected. The output check rejects an answer that says the information is unavailable while `needs_escalation` is `false`.
[`docs/architecture.md`](docs/architecture.md) has the diagrams, a glossary and a one-question trace.

## Status: what is tested and what is not

Everything below was run on Windows 11 with Python 3.12, CPU only. Nothing was run on macOS or Linux, and Python 3.10 and 3.11 were not tried.
The code uses `pathlib` and ASCII console output to stay portable, but that is unverified on other platforms.

| Part | Tested | Not tested |
|---|---|---|
| `rag-starter` | 288 tests pass offline with the optional `pypdf` installed (1 skipped; without `pypdf` the count is 286 passed and 3 skipped). The Start here commands, the access-control demo and the retrieval eval were run and the README output matches. The optional sentence-transformers embedder and cross-encoder re-ranker were each tried once by hand. | A live call to the Anthropic API (no API key was available; `rag/llm.py` is tested with a fake client only). |
| `finetune-starter` | 79 tests pass with torch, transformers, peft and nbformat installed (78 pass and 1 is skipped without nbformat). Smoke runs with tiny random models. Complete train-and-evaluate runs on `Qwen/Qwen2.5-0.5B-Instruct` (defaults) and `HuggingFaceTB/SmolLM2-135M-Instruct` (4 and 8 epochs), with the numbers above; the two sets of commands in the quick start were re-run from the repository root and gave the same counts. The notebook's code cells were executed locally. | Any GPU, Colab itself, other seeds, other library versions. |
| `hybrid` | 40 tests. `--dry-run` in a fresh environment with no torch, transformers or peft. The live path with two small real adapters (SmolLM2-135M and the default Qwen2.5-0.5B, three questions each, described above) and its failure path with a tiny random model. | Any GPU, more than three questions, and any measure of answer quality. |
| `docs` | The three Mermaid diagrams in `docs/architecture.md` parse without errors in the Mermaid 12.0.0 parser (checked in Node). The PDF's text was extracted and read. | Viewing the diagrams rendered on GitHub or in your editor; reading the PDF in every PDF viewer. |

Small evaluation sets in this repository (18 retrieval questions, 24 validation prompts, 10 probe prompts) move in big steps, so
treat their results as a smoke signal, not evidence. Build a set from your own documents before you tune anything.

## Run the tests

From the repository root, with the environment from "Start here" active (`pip install -r rag-starter/requirements-dev.txt` adds pytest; one fine-tune test is skipped unless `nbformat` is installed):

```powershell
python -m pytest rag-starter
python -m pytest finetune-starter/tests
python -m pytest hybrid
```

Generated folders (`rag-starter/index`, `finetune-starter/outputs`, `finetune-starter/results.json`) are ignored by git.

## License and disclaimer

MIT. See [`LICENSE`](LICENSE).

This is teaching material, not a production system and not legal, security or compliance advice. The guidance is illustrative:
verify it on your own data and your own stack. The permission-filter demo shows a mechanism, not a security audit. Check the
license and usage terms of any model you download, and never put personal, customer or otherwise sensitive data into training files.

## References

- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* arXiv:2005.11401. https://arxiv.org/abs/2005.11401
- Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685. https://arxiv.org/abs/2106.09685
- Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314. https://arxiv.org/abs/2305.14314
- Hugging Face PEFT documentation: https://huggingface.co/docs/peft
