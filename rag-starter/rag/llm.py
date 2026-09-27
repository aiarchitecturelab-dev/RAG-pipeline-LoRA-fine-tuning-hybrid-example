"""Step 8 of the pipeline: ask the model. The ONLY module that needs an API key.

This is a thin wrapper around the official Anthropic SDK. Things worth knowing,
because they are easy to get wrong with current models:

* The client reads ANTHROPIC_API_KEY from the environment: anthropic.Anthropic().
* Model ids are exact strings with no date suffix. The default is
  "claude-opus-5". Cheaper models ("claude-sonnet-5", "claude-haiku-4-5") are
  your choice: set LLM_MODEL and compare answer quality on your own questions.
* Do NOT pass temperature, top_p or top_k, do NOT pre-fill the assistant turn,
  and do NOT send a thinking budget (budget_tokens). According to Anthropic's
  documentation at the time of writing, Opus 5 and Sonnet 5 class models reject
  all of these with HTTP 400. Haiku 4.5 behaves differently, so check the docs
  for the model you choose. Ask for the format you want in the prompt instead.
* Opus-class models think adaptively by default, and thinking tokens count
  against max_tokens. See MAX_TOKENS below for why it is 16000.
* response.content is a LIST of blocks. It may contain "thinking" blocks next
  to the "text" blocks, so never read content[0].text: join the text blocks.
* Always look at response.stop_reason: "refusal" means the model declined,
  "max_tokens" means the answer was cut off.
* The SDK already retries transient errors (rate limits, 5xx, dropped
  connections) with backoff. We add no retry loop of our own.

Testing: no live call is made by the test-suite. generate_answer() accepts a
`client` argument so tests can pass a fake object with the same shape.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import anthropic

DEFAULT_MODEL = "claude-opus-5"

# Upper bound on everything the model generates for one request: the visible
# answer AND any thinking. Opus-class models use adaptive thinking by default
# and those thinking tokens count against max_tokens, so a small limit (such
# as 2000) can be used up by thinking and leave a truncated or empty answer.
# 16000 is a generous ceiling for a plain (non-streaming) request. It is only a
# limit, not a target: short answers cost only the tokens they use. Much larger
# limits are a reason to switch to streaming, which this template does not do.
MAX_TOKENS = 16000

REFUSAL_MESSAGE = (
    "The model declined to answer this request. "
    "Try rephrasing the question, or ask a colleague or the People Team."
)


class LLMError(RuntimeError):
    """A friendly, printable description of why the model call failed."""


@dataclass
class LLMResult:
    """The model's answer plus anything the caller should know about it."""

    text: str
    stop_reason: Optional[str]
    model: str
    warnings: list[str] = field(default_factory=list)


def extract_text(content_blocks: Any) -> str:
    """Join the text of every block whose type is "text".

    Other block types (for example "thinking") are skipped. We join with an
    empty string because when the API splits one answer into several text
    blocks it does so mid-sentence.
    """
    parts = []
    for block in content_blocks or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts).strip()


def generate_answer(
    system: str,
    user: str,
    client: Optional[Any] = None,
    model: Optional[str] = None,
) -> LLMResult:
    """Send one request and return the answer, or raise LLMError.

    `client` is only for tests and advanced use; normally leave it None and the
    function creates anthropic.Anthropic() (which reads ANTHROPIC_API_KEY).
    """
    model_id = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
    if client is None:
        client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model=model_id,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    # Most specific exception first: AuthenticationError, RateLimitError and
    # BadRequestError are all subclasses of APIStatusError, so APIStatusError
    # must come after them or it would swallow them.
    except anthropic.AuthenticationError as exc:
        raise LLMError(
            "Authentication failed: the API key was rejected. "
            "Check ANTHROPIC_API_KEY."
        ) from exc
    except anthropic.RateLimitError as exc:
        raise LLMError(
            "Rate limit reached even after the SDK's automatic retries. "
            "Wait a little and try again."
        ) from exc
    except anthropic.BadRequestError as exc:
        raise LLMError(
            f"The API rejected the request as invalid: {exc.message}. "
            f"Check that LLM_MODEL={model_id!r} is a valid model id."
        ) from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(
            f"The API returned an error (HTTP {exc.status_code}): {exc.message}"
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise LLMError(
            "Could not reach the API. Check your internet connection and proxy settings."
        ) from exc
    except TypeError as exc:
        # The SDK raises TypeError("Could not resolve authentication method...")
        # when it finds no credentials at all. Anything else is a real bug.
        if "authentication" in str(exc).lower():
            raise LLMError(
                "No API credentials found. Set the ANTHROPIC_API_KEY environment "
                "variable (see .env.example)."
            ) from exc
        raise

    stop_reason = getattr(response, "stop_reason", None)
    warnings: list[str] = []

    if stop_reason == "refusal":
        return LLMResult(text=REFUSAL_MESSAGE, stop_reason=stop_reason, model=model_id)

    text = extract_text(getattr(response, "content", None))
    if stop_reason == "max_tokens":
        warnings.append(
            f"The answer was cut off because it reached max_tokens ({MAX_TOKENS}). "
            "Raise MAX_TOKENS in rag/llm.py or ask a narrower question."
        )
    if not text:
        warnings.append("The model returned no text.")
    return LLMResult(text=text, stop_reason=stop_reason, model=model_id, warnings=warnings)
