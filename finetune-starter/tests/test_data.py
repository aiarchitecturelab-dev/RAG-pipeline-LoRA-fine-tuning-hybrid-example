"""Checks on the data files. No downloads and no torch needed."""
import json
import re
from pathlib import Path

import pytest

from common import REQUIRED_KEYS, check_schema, read_jsonl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAG_DOCS = ROOT.parent / "rag-starter" / "data" / "docs"  # sibling starter; tests skip if it is absent
ID_RE = re.compile(r"^\[([A-Z]+-\d+)\] ", re.MULTILINE)
NUM_RE = re.compile(r"\d[\d,.:]*\d|\d")
UNANSWERABLE_PHRASE = "not available in the provided context"

# Subjects that the rag-starter documents state facts about. The fine-tuning data is about a
# DIFFERENT fictional company ("Globex Corp") and must not touch these subjects at all, so nothing
# the adapter memorizes can agree or clash with a retrieved fact. This is a keyword screen, not a proof.
RAG_SUBJECTS = {
    "parental leave": r"parental|maternity|paternity|adoption|newborn|new baby|birth of",
    "working hours and remote work": r"remote|work(?:ing)? from home|from another country|core (?:working |collaboration )?hours|working week",
    "annual leave": r"annual leave|vacation|paid time off|days off",
    "sick leave": r"\bsick|medical certificate|doctor",
    "performance and salary reviews": r"performance review|salary review|pay rise|pay raise|\brating",
    "notice periods": r"notice period|resign|notice to end",
    "raising concerns": r"whistle|hotline|speak up|retaliat|anonymous",
    "expenses and spending limits": r"hotel|\bmeals?\b|per diem|taxi|ride share|expense claim|expense polic|itemi[sz]ed",
    "client entertainment and gifts": r"entertain|hospitality|\bgifts?\b",
    "air travel and travel booking": r"business class|economy|flight|rental car|travel portal|airline",
    "passwords and MFA": r"password|multi-factor|\bmfa\b|\bvpn\b|one-time code",
    "device and account security": r"lost or stolen|stolen|screen lock|inactiv|phishing|security incident|data classification|\busb\b|removable|encrypt|public wi-?fi",
    "AI tools": r"\bai\b|artificial intelligence|chatbot",
    "executive compensation": r"executive|severance|clawback|annual bonus|target bonus|base salar|salary band|compensation|pension|long-term incentive",
}
BRITISH = re.compile(
    r"\b(?:itemised|anonymised|enrolment|kilometre|authorised|catalogue|recognised|programme|colour|behaviour|centre|licence)\b",
    re.IGNORECASE,
)


@pytest.fixture(scope="module")
def train():
    return read_jsonl(DATA / "train.jsonl")


@pytest.fixture(scope="module")
def val():
    return read_jsonl(DATA / "val.jsonl")


@pytest.fixture(scope="module")
def probe():
    return read_jsonl(DATA / "general_probe.jsonl")


def parts(row):
    msgs = row["messages"]
    return msgs[0]["content"], msgs[1]["content"], json.loads(msgs[2]["content"])


def context_ids(user_text):
    return ID_RE.findall(user_text)


def passages(user_text):
    """The passage lines ('[ID] Title: text') of one prompt."""
    context = user_text.split("\n\nQUESTION: ", 1)[0]
    return context.splitlines()[1:]


def test_files_parse_and_have_expected_size(train, val, probe):
    assert 60 <= len(train) <= 80
    assert 20 <= len(val) <= 30
    assert len(probe) == 10


def test_chat_format(train, val):
    for row in train + val:
        assert set(row) == {"messages"}
        roles = [m["role"] for m in row["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert all(isinstance(m["content"], str) and m["content"].strip() for m in row["messages"])


def test_user_turn_has_context_and_question(train, val):
    for row in train + val:
        _, user, _ = parts(row)
        assert user.startswith("CONTEXT:\n")
        assert "\n\nQUESTION: " in user
        assert context_ids(user), "every context passage must carry an [ID]"


def test_every_assistant_message_is_valid_json_with_required_keys(train, val):
    for row in train + val:
        raw = row["messages"][2]["content"]
        obj = json.loads(raw)
        assert list(obj) == list(REQUIRED_KEYS), "keys must be exactly the required keys, in order"
        assert check_schema(obj) == []
        assert obj["tone"] == "formal"
        assert isinstance(obj["sources"], list)
        assert isinstance(obj["needs_escalation"], bool)


def test_answers_are_two_or_three_sentences(train, val):
    for row in train + val:
        answer = parts(row)[2]["answer"]
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", answer.strip())
        assert 2 <= len(sentences) <= 3, answer
        assert answer.strip().endswith("."), answer


def test_sources_are_ids_present_in_the_context(train, val):
    for row in train + val:
        _, user, obj = parts(row)
        ids = set(context_ids(user))
        assert set(obj["sources"]) <= ids
        assert len(set(obj["sources"])) == len(obj["sources"])


def test_unanswerable_examples_follow_the_escalation_contract(train, val):
    for row in train + val:
        _, _, obj = parts(row)
        if obj["needs_escalation"]:
            assert UNANSWERABLE_PHRASE in obj["answer"]
            assert obj["sources"] == []
        else:
            assert UNANSWERABLE_PHRASE not in obj["answer"]
            assert obj["sources"], "answerable examples must cite at least one source"


def test_unanswerable_share_is_about_a_quarter_to_a_third(train, val):
    """The first version had about 15 percent; a default-model run never learned to escalate, so the share was raised."""
    for rows in (train, val):
        share = sum(parts(r)[2]["needs_escalation"] for r in rows) / len(rows)
        assert 0.25 <= share <= 0.30, share


def test_val_has_enough_unanswerable_examples_to_measure_escalation(val):
    assert sum(parts(r)[2]["needs_escalation"] for r in val) >= 6


def test_smoke_slices_contain_an_unanswerable_example(train, val):
    """--smoke and evaluate.py --limit use the first rows, so they must include an escalation case."""
    assert any(parts(r)[2]["needs_escalation"] for r in train[:8])
    assert any(parts(r)[2]["needs_escalation"] for r in val[:4])


def test_answers_do_not_invent_numbers(train, val):
    """Every number in an answer must appear in the context or in the question."""
    for row in train + val:
        _, user, obj = parts(row)
        for num in NUM_RE.findall(obj["answer"]):
            assert num.strip(".,:") in user, f"number {num!r} not found in the prompt: {obj['answer']}"


def test_train_and_val_do_not_overlap(train, val):
    train_users = {parts(r)[1] for r in train}
    val_users = {parts(r)[1] for r in val}
    assert not train_users & val_users, "identical prompts in train and val"
    train_questions = {u.split("QUESTION: ", 1)[1] for u in train_users}
    val_questions = {u.split("QUESTION: ", 1)[1] for u in val_users}
    assert not train_questions & val_questions, "identical questions in train and val"
    train_ids = {i for u in train_users for i in context_ids(u)}
    val_ids = {i for u in val_users for i in context_ids(u)}
    assert not train_ids & val_ids, "val must use passages the model never saw in training"
    # the same passage text must not hide under another id either
    train_texts = {ID_RE.sub("", p) for u in train_users for p in passages(u)}
    val_texts = {ID_RE.sub("", p) for u in val_users for p in passages(u)}
    assert not train_texts & val_texts, "the same passage text appears in train and val"


def test_no_duplicate_prompts_inside_train(train):
    users = [parts(r)[1] for r in train]
    assert len(users) == len(set(users))


def test_same_passage_id_always_has_same_text(train, val):
    seen = {}
    for row in train + val:
        _, user, _ = parts(row)
        for line in passages(user):
            match = ID_RE.match(line)
            assert match, line
            pid = match.group(1)
            assert seen.setdefault(pid, line) == line, f"passage {pid} appears with two different texts"


def test_no_personal_data_patterns(train, val, probe):
    blob = json.dumps(train + val + probe)
    assert "@" not in blob
    assert "http" not in blob.lower()
    assert not re.search(r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b", blob), "looks like a phone number"


def test_general_probe_format(probe):
    for row in probe:
        assert set(row) == {"prompt", "expected"}
        assert isinstance(row["prompt"], str) and row["prompt"].strip()
        expected = row["expected"]
        assert isinstance(expected, str) or (
            isinstance(expected, list) and expected and all(isinstance(e, str) for e in expected)
        )
    assert len({r["prompt"] for r in probe}) == len(probe)


# ------------------------------------------------ consistency with the RAG corpus
def test_data_is_about_globex_and_never_mentions_the_rag_company(train, val, probe):
    for row in train + val:
        assert "Globex Corp" in row["messages"][0]["content"]
    blob = json.dumps(train + val + probe)
    assert "acme" not in blob.lower(), "the fine-tuning data must use a different fictional company than the RAG corpus"
    assert "Globex Corp" in blob


def test_no_parental_leave_example_so_the_demo_answer_cannot_be_memorized():
    """The video demo question is 'What's our parental leave policy?': no data file may touch the topic."""
    raw = "\n".join(
        (DATA / name).read_text(encoding="utf-8") for name in ("train.jsonl", "val.jsonl", "general_probe.jsonl")
    )
    assert not re.search(RAG_SUBJECTS["parental leave"], raw, re.IGNORECASE)


def test_no_example_touches_a_subject_the_rag_documents_cover(train, val, probe):
    problems = []
    for split, rows in (("train", train), ("val", val)):
        for n, row in enumerate(rows):
            text = " ".join(m["content"] for m in row["messages"][1:])
            for subject, pattern in RAG_SUBJECTS.items():
                if re.search(pattern, text, re.IGNORECASE):
                    problems.append(f"{split}[{n}] mentions {subject!r}")
    for row in probe:
        for subject, pattern in RAG_SUBJECTS.items():
            if re.search(pattern, row["prompt"], re.IGNORECASE):
                problems.append(f"probe {row['prompt']!r} mentions {subject!r}")
    assert not problems, problems


def test_no_passage_title_or_sentence_is_copied_from_the_rag_documents(train, val):
    if not RAG_DOCS.is_dir():
        pytest.skip("rag-starter/data/docs not found next to this folder")
    headings, sentences = set(), set()
    for doc in RAG_DOCS.glob("*.md"):
        text = doc.read_text(encoding="utf-8")
        headings |= {h.strip().lower() for h in re.findall(r"^#{1,3}\s+(.+)$", text, re.MULTILINE)}
        sentences |= {" ".join(x.lower().split()) for x in re.split(r"(?<=[.!?])\s+", text) if len(x) > 30}
    assert headings, "no headings found in the RAG documents"
    for row in train + val:
        for line in passages(parts(row)[1]):
            body = re.sub(r"^\[[A-Z]+-\d+\] ", "", line)
            title = body.split(":", 1)[0].strip().lower()
            assert title not in headings, f"passage title {title!r} equals a RAG section heading"
            for sentence in re.split(r"(?<=[.!?])\s+", body):
                assert " ".join(sentence.lower().split()) not in sentences, sentence


def test_american_spelling_in_the_data(train, val, probe):
    blob = json.dumps(train + val + probe)
    assert not BRITISH.findall(blob), BRITISH.findall(blob)
