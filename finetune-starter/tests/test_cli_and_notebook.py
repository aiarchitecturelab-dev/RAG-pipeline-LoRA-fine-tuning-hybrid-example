"""Argument parsers, start-up checks and notebook checks (no torch, no downloads)."""
import ast
import json
import re
import sys
from pathlib import Path

import pytest

import common
import evaluate
import merge_adapter
import train_lora

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "lora_finetune.ipynb"
HEAVY = {"torch", "transformers", "peft", "datasets"}


def make_adapter(folder: Path, base_model="some-org/some-base-model"):
    folder.mkdir(parents=True, exist_ok=True)
    config = {"peft_type": "LORA"}
    if base_model is not None:
        config["base_model_name_or_path"] = base_model
    (folder / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    return folder


# ---------------------------------------------------------------- train_lora.py
def test_train_parser_has_all_required_arguments_and_defaults():
    args = train_lora.build_parser().parse_args([])
    assert args.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert Path(args.data_dir) == ROOT / "data"
    for name in ("out_dir", "epochs", "lr", "rank", "alpha", "dropout", "batch_size",
                 "grad_accum", "max_len", "max_steps", "seed", "smoke"):
        assert hasattr(args, name), name
    assert args.smoke is False and args.max_steps == -1


def test_train_parser_accepts_overrides():
    args = train_lora.build_parser().parse_args(
        ["--model", "tiny/model", "--epochs", "2", "--lr", "1e-4", "--rank", "8", "--alpha", "16",
         "--dropout", "0.1", "--batch-size", "2", "--grad-accum", "1", "--max-steps", "3",
         "--seed", "7", "--smoke"]
    )
    assert (args.model, args.epochs, args.lr, args.rank, args.alpha) == ("tiny/model", 2.0, 1e-4, 8, 16.0)
    assert (args.dropout, args.batch_size, args.grad_accum, args.max_steps, args.seed) == (0.1, 2, 1, 3, 7)
    assert args.smoke is True


def test_smoke_help_describes_what_smoke_really_does():
    text = " ".join(train_lora.build_parser().format_help().split())
    assert "3 optimizer steps" in text and "meaningless" in text
    assert "1 epoch" not in text


def test_smoke_runs_get_no_interpretive_overfitting_hint():
    rising = [
        {"epoch": 1, "train_loss": 2.0, "val_loss": 1.5},
        {"epoch": 2, "train_loss": 1.0, "val_loss": 1.2},
        {"epoch": 3, "train_loss": 0.4, "val_loss": 1.4},
    ]
    assert "overfitting" in common.training_remark(rising, smoke=False)
    remark = common.training_remark(rising, smoke=True)
    assert remark == "smoke run: losses are meaningless (plumbing check only)"
    assert "overfitting" not in remark


# ------------------------------------------------------------------ evaluate.py
def test_evaluate_parser_defaults():
    args = evaluate.build_parser().parse_args([])
    assert args.adapter is None and args.prompt_baseline is False
    assert args.model is None, "--model must default to None so the adapter's base model can be used"
    assert Path(args.data_dir) == ROOT / "data"
    assert "(default: None)" not in evaluate.build_parser().format_help()


def test_base_model_is_read_from_the_adapter_when_model_is_omitted(tmp_path):
    adapter = make_adapter(tmp_path / "adapter", "org/recorded-model")
    assert evaluate.resolve_base_model(None, str(adapter)) == ("org/recorded-model", [])


def test_explicit_model_that_differs_from_the_adapter_gives_a_warning(tmp_path):
    adapter = make_adapter(tmp_path / "adapter", "org/recorded-model")
    name, warnings = evaluate.resolve_base_model("org/other-model", str(adapter))
    assert name == "org/other-model"
    assert len(warnings) == 1 and "org/recorded-model" in warnings[0] and "differs" in warnings[0]
    assert evaluate.resolve_base_model("org/recorded-model", str(adapter)) == ("org/recorded-model", [])


def test_without_an_adapter_the_default_model_is_used():
    assert evaluate.resolve_base_model(None, None) == ("Qwen/Qwen2.5-0.5B-Instruct", [])
    assert evaluate.resolve_base_model("my/model", None) == ("my/model", [])


def test_adapter_config_without_a_base_model_needs_an_explicit_model(tmp_path):
    adapter = make_adapter(tmp_path / "adapter", base_model=None)
    with pytest.raises(SystemExit, match="does not record a base model"):
        evaluate.resolve_base_model(None, str(adapter))
    assert evaluate.resolve_base_model("my/model", str(adapter)) == ("my/model", [])


def test_missing_adapter_exits_with_one_clear_line_before_anything_is_loaded(tmp_path):
    missing = str(tmp_path / "nope")
    with pytest.raises(SystemExit) as info:
        evaluate.main(["--adapter", missing])
    message = str(info.value)
    assert message == f"no adapter at {missing}; run train_lora.py first (a smoke run writes to outputs/smoke)"
    assert "\n" not in message
    # a folder that exists but has no adapter_config.json is just as wrong
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no adapter at"):
        evaluate.main(["--adapter", str(empty)])


def test_merge_adapter_also_checks_the_adapter_first(tmp_path):
    with pytest.raises(SystemExit, match="no adapter at"):
        merge_adapter.main(["--adapter", str(tmp_path / "nope")])


# ------------------------------------------------------ missing-dependency messages
def test_exit_missing_dependency_is_one_line_with_the_install_hint():
    with pytest.raises(SystemExit) as info:
        common.exit_missing_dependency(ImportError("No module named 'torch'", name="torch"))
    message = str(info.value)
    assert "torch" in message and "pip install -r requirements.txt" in message and "\n" not in message


@pytest.mark.parametrize("blocked", ["torch", "peft"])
def test_scripts_turn_a_missing_library_into_the_install_hint(monkeypatch, tmp_path, blocked):
    """Setting sys.modules[name] = None makes `import name` raise ImportError, exactly like an absent package."""
    monkeypatch.setitem(sys.modules, blocked, None)
    adapter = make_adapter(tmp_path / "adapter")
    calls = {
        "torch": [
            lambda: train_lora.main(["--smoke"]),
            lambda: evaluate.main(["--limit", "1"]),
        ],
        "peft": [
            lambda: evaluate.main(["--limit", "1", "--adapter", str(adapter)]),
            lambda: merge_adapter.main(["--adapter", str(adapter)]),
        ],
    }[blocked]
    for call in calls:
        with pytest.raises(SystemExit) as info:
            call()
        message = str(info.value)
        assert "pip install -r requirements.txt" in message and "\n" not in message
        # without torch installed, torch is the first import to fail; either way the package is named
        assert blocked in message or "torch" in message


def _heavy_imports_outside_try(path: Path):
    """Heavy imports that would fail without a readable message.

    Two rules: (1) no heavy import at module level (so --help and the tests work without torch);
    (2) inside main() - the entry point, where the first heavy import happens - every heavy import
    sits in a try block that catches ImportError. Helper functions such as generate_texts import
    torch lazily, but they are only reached after main() has already imported it successfully.
    """
    offenders = []

    def heavy(node):
        if isinstance(node, ast.Import):
            names = {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names = {(node.module or "").split(".")[0]}
        else:
            names = set()
        return bool(names & HEAVY)

    def visit(node, guarded):
        if isinstance(node, ast.Try):
            catches = any(h.type is not None and "ImportError" in ast.dump(h.type) for h in node.handlers)
            for child in node.body:
                visit(child, guarded or catches)
            for part in (node.handlers, node.orelse, node.finalbody):
                for child in part:
                    visit(child, guarded)
            return
        if heavy(node) and not guarded:
            offenders.append(f"{path.name}:{node.lineno}")
        for child in ast.iter_child_nodes(node):
            visit(child, guarded)

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            visit(node, False)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            continue  # lazy imports in helpers: see the docstring
        elif heavy(node):
            offenders.append(f"{path.name}:{node.lineno} (module level)")
    return offenders


@pytest.mark.parametrize("script", ["train_lora.py", "evaluate.py", "merge_adapter.py"])
def test_heavy_imports_are_wrapped_in_try_except_importerror(script):
    assert _heavy_imports_outside_try(ROOT / script) == []


def test_the_import_guard_check_itself_catches_an_unguarded_import(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def main():\n    import torch\n", encoding="utf-8")
    assert _heavy_imports_outside_try(bad)
    good = tmp_path / "good.py"
    good.write_text("def main():\n    try:\n        import torch\n    except ImportError:\n        pass\n", encoding="utf-8")
    assert _heavy_imports_outside_try(good) == []


# ------------------------------------------------------------------- the notebook
def test_notebook_is_wellformed_nbformat_v4_with_empty_outputs():
    raw = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert raw["nbformat"] == 4
    code_cells = [c for c in raw["cells"] if c["cell_type"] == "code"]
    assert code_cells and any(c["cell_type"] == "markdown" for c in raw["cells"])
    for cell in code_cells:
        assert cell["outputs"] == [] and cell["execution_count"] is None
    nbformat = pytest.importorskip("nbformat")  # only needed for this validation; install with: pip install nbformat
    nb = nbformat.read(str(NOTEBOOK), as_version=4)
    nbformat.validate(nb)


def notebook_source() -> str:
    raw = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in raw["cells"])


def notebook_probe_pass():
    """Compile the notebook's own probe_pass so the test exercises the code a user would run."""
    raw = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in raw["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "def probe_pass" not in source:
            continue  # other cells hold %pip magics and cannot be parsed as plain Python
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name == "probe_pass":
                namespace = {"re": re}
                exec(ast.get_source_segment(source, node), namespace)  # noqa: S102 - our own notebook code
                return namespace["probe_pass"]
    raise AssertionError("probe_pass not found in the notebook")


PROBE_CASES = [
    ("The capital is PARIS.", "Paris"),
    ("I do not know.", "Paris"),
    ("There are 7 days.", ["7", "seven"]),
    ("There are seven days.", ["7", "seven"]),
    ("There are 17 days.", "7"),
    ("There are 17 days.", ["7", "seven"]),
    ("12 + 15 = 27", "27"),
    ("The answer is 127.", "27"),
    ("The answer is 27.5", "27"),
    ("Photosynthesis uses LIGHT.", "light"),
    ("", "January"),
]


@pytest.mark.parametrize("text, expected", PROBE_CASES)
def test_notebook_probe_pass_agrees_with_common(text, expected):
    assert notebook_probe_pass()(text, expected) == common.probe_pass(text, expected)


def test_notebook_probe_pass_applies_the_digit_boundary_rule():
    probe_pass = notebook_probe_pass()
    assert not probe_pass("There are 17 days.", "7")
    assert probe_pass("There are 7 days.", "7")


def test_notebook_and_readme_have_no_placeholder_colab_url_and_no_old_company():
    text = notebook_source() + (ROOT / "README.md").read_text(encoding="utf-8")
    assert "<your-user>" not in text and "<your-repo>" not in text
    assert "colab.research.google.com/github" not in text
    assert "File > Upload notebook" in notebook_source()
    assert "acme" not in notebook_source().lower()
    assert "Globex Corp" in notebook_source()
