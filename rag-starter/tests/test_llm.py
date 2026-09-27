"""llm.py with a FAKE client. No network, no API key, no real API call.

These tests check how OUR code reads responses and shapes requests. They cannot
show that the live API accepts the request; that needs a real key.
"""

from __future__ import annotations

import pytest

import anthropic

from fakes import FakeClient, fake_response, text_block, thinking_block
from rag import llm
from rag.config import Config
from rag.llm import LLMError, extract_text, generate_answer
from rag.pipeline import RagPipeline

try:  # the SDK's HTTP library is called httpx2 in anthropic 1.x, httpx before that
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx


def status_error(error_class, status_code: int, message: str = "boom"):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    return error_class(message, response=response, body=None)


# ------------------------------------------------------------ text extraction


def test_text_is_extracted_from_mixed_thinking_and_text_blocks():
    blocks = [
        thinking_block("private reasoning, must not leak"),
        text_block("Twelve weeks "),
        thinking_block("more reasoning"),
        text_block("[Source: hr-policy-2026.md, Parental leave]."),
    ]
    assert extract_text(blocks) == "Twelve weeks [Source: hr-policy-2026.md, Parental leave]."


def test_extract_text_ignores_blocks_without_text_and_handles_empty_content():
    assert extract_text([thinking_block()]) == ""
    assert extract_text([]) == ""
    assert extract_text(None) == ""


def test_generate_answer_returns_the_joined_text():
    client = FakeClient(fake_response([thinking_block(), text_block("Hello "), text_block("world")]))
    result = generate_answer("sys", "user question", client=client)
    assert result.text == "Hello world"
    assert result.stop_reason == "end_turn"
    assert result.warnings == []


# ------------------------------------------------------------ request shape


def test_request_uses_exactly_the_documented_arguments(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    client = FakeClient(fake_response([text_block("ok")]))
    generate_answer("SYSTEM TEXT", "USER TEXT", client=client)

    (call,) = client.calls
    assert set(call) == {"model", "max_tokens", "system", "messages"}
    assert call["model"] == "claude-opus-5"
    assert call["max_tokens"] == 16000
    assert call["system"] == "SYSTEM TEXT"
    assert call["messages"] == [{"role": "user", "content": "USER TEXT"}]


def test_request_has_no_sampling_parameters_thinking_budget_or_prefill():
    client = FakeClient(fake_response([text_block("ok")]))
    generate_answer("s", "u", client=client)
    (call,) = client.calls
    for forbidden in ("temperature", "top_p", "top_k", "thinking"):
        assert forbidden not in call
    assert call["messages"][-1]["role"] == "user"  # no assistant prefill


def test_max_tokens_leaves_room_for_adaptive_thinking():
    """Thinking tokens count against max_tokens, so 2000 was too tight for Opus-class models."""
    assert llm.MAX_TOKENS >= 16000
    client = FakeClient(fake_response([text_block("ok")]))
    generate_answer("s", "u", client=client)
    assert client.calls[0]["max_tokens"] == llm.MAX_TOKENS


def test_max_tokens_warning_names_the_current_limit():
    client = FakeClient(fake_response([text_block("cut")], stop_reason="max_tokens"))
    result = generate_answer("s", "u", client=client)
    assert any(str(llm.MAX_TOKENS) in warning for warning in result.warnings)


def test_request_does_not_enable_the_beta_refusal_fallbacks():
    """The plain client.messages.create call is kept: no betas, no fallbacks."""
    client = FakeClient(fake_response([text_block("ok")]))
    generate_answer("s", "u", client=client)
    (call,) = client.calls
    for forbidden in ("betas", "fallbacks", "extra_headers", "extra_body"):
        assert forbidden not in call


def test_model_can_be_chosen_with_the_environment_variable(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5")
    client = FakeClient(fake_response([text_block("ok")]))
    result = generate_answer("s", "u", client=client)
    assert client.calls[0]["model"] == "claude-sonnet-5"
    assert result.model == "claude-sonnet-5"


def test_default_model_id_has_no_date_suffix():
    assert llm.DEFAULT_MODEL == "claude-opus-5"


# ------------------------------------------------------------ stop reasons


def test_refusal_gives_a_friendly_message_and_no_exception():
    client = FakeClient(fake_response([], stop_reason="refusal"))
    result = generate_answer("s", "u", client=client)
    assert result.stop_reason == "refusal"
    assert result.text == llm.REFUSAL_MESSAGE
    assert "declined" in result.text


def test_refusal_ignores_any_partial_text_in_the_response():
    client = FakeClient(fake_response([text_block("partial")], stop_reason="refusal"))
    assert generate_answer("s", "u", client=client).text == llm.REFUSAL_MESSAGE


def test_max_tokens_adds_a_warning_and_keeps_the_partial_text():
    client = FakeClient(fake_response([text_block("The answer starts and then")], stop_reason="max_tokens"))
    result = generate_answer("s", "u", client=client)
    assert result.text == "The answer starts and then"
    assert any("max_tokens" in warning for warning in result.warnings)


def test_empty_answer_adds_a_warning():
    client = FakeClient(fake_response([thinking_block()], stop_reason="end_turn"))
    result = generate_answer("s", "u", client=client)
    assert result.text == ""
    assert any("no text" in warning for warning in result.warnings)


# ---------------------------------------------------------- error handling


@pytest.mark.parametrize(
    "error, expected_fragment",
    [
        (status_error(anthropic.AuthenticationError, 401), "Authentication failed"),
        (status_error(anthropic.RateLimitError, 429), "Rate limit"),
        (status_error(anthropic.BadRequestError, 400, "model: not found"), "invalid"),
        (status_error(anthropic.InternalServerError, 500, "server exploded"), "HTTP 500"),
        (
            anthropic.APIConnectionError(
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            ),
            "Could not reach the API",
        ),
    ],
)
def test_api_errors_become_friendly_llm_errors(error, expected_fragment):
    client = FakeClient(error=error)
    with pytest.raises(LLMError, match=expected_fragment) as excinfo:
        generate_answer("s", "u", client=client)
    assert excinfo.value.__cause__ is error  # the original exception is kept for debugging


def test_missing_credentials_are_explained():
    error = TypeError(
        "Could not resolve authentication method. Expected one of api_key, auth_token, "
        "or credentials to be set."
    )
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        generate_answer("s", "u", client=FakeClient(error=error))


def test_unrelated_type_errors_are_not_swallowed():
    with pytest.raises(TypeError, match="something else"):
        generate_answer("s", "u", client=FakeClient(error=TypeError("something else")))


def test_bad_request_message_names_the_model_id():
    error = status_error(anthropic.BadRequestError, 400, "unknown model")
    with pytest.raises(LLMError, match="claude-opus-5"):
        generate_answer("s", "u", client=FakeClient(error=error), model="claude-opus-5")


# ------------------------------------------------- pipeline + fake client


def _pipeline_with_fake_answer(pipeline: RagPipeline, answer_text: str, stop_reason: str = "end_turn"):
    client = FakeClient(fake_response([thinking_block(), text_block(answer_text)], stop_reason))
    prepared = pipeline.prepare("What's our parental leave policy?", "employee")
    return pipeline.answer(prepared, client=client), client


def test_pipeline_validates_the_citations_of_the_llm_answer(pipeline):
    answer, client = _pipeline_with_fake_answer(
        pipeline,
        "You get 12 weeks of paid leave [Source: hr-policy-2026.md, Parental leave].",
    )
    assert answer.called_llm
    assert answer.citations.ok
    assert client.calls[0]["system"] == answer.prepared.prompt.system
    assert client.calls[0]["messages"][0]["content"] == answer.prepared.prompt.user


def test_pipeline_flags_a_citation_to_a_source_that_was_never_retrieved(pipeline):
    answer, _ = _pipeline_with_fake_answer(
        pipeline,
        "Executives get more [Source: exec-compensation-2026.md, Annual bonus].",
    )
    assert not answer.citations.ok
    assert [c.source for c in answer.citations.unknown] == ["exec-compensation-2026.md"]


def test_pipeline_does_not_validate_citations_of_a_refusal(pipeline):
    answer, _ = _pipeline_with_fake_answer(pipeline, "", stop_reason="refusal")
    assert answer.text == llm.REFUSAL_MESSAGE
    assert answer.citations is None


def test_pipeline_does_not_call_the_llm_when_nothing_relevant_is_allowed(pipeline):
    client = FakeClient(error=AssertionError("the LLM must not be called"))
    # A question made only of stop words has an all-zero vector, so every score is 0.
    prepared = pipeline.prepare("what is the", "employee")
    assert prepared.hits == []
    answer = pipeline.answer(prepared, client=client)
    assert not answer.called_llm
    assert answer.text.startswith("I do not know")
    assert client.calls == []


def test_config_default_matches_the_llm_default():
    assert Config().llm_model == llm.DEFAULT_MODEL
