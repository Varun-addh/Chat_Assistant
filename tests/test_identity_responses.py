"""Regression tests for identity/ownership wording.

Goal: identity guard should be:
- deterministic
- concise
- accurate: Stratax is an application/platform using AI model APIs
- avoid risky/misleading phrases

We intentionally call the internal identity helper to avoid external LLM calls.
"""

from app.services.llm_service import LLMService


def test_identity_who_are_you_mentions_application_and_models() -> None:
    svc = LLMService()
    text = svc._identity_response_text("who are you?").lower()
    assert "application" in text or "platform" in text
    assert "ai language model" in text or "language model" in text


def test_identity_chatgpt_does_not_claim_independence_or_codebase() -> None:
    svc = LLMService()
    text = svc._identity_response_text("are you chatgpt?").lower()

    # Should not use risky marketing/legal phrasing
    forbidden = [
        "independent of openai",
        "own codebase",
        "i am not chatgpt",  # we prefer positive framing: "I'm Stratax AI"
    ]
    for f in forbidden:
        assert f not in text

    # Should still clarify separation
    assert "chatgpt" in text
    assert "openai" in text


def test_identity_google_gemini_no_google_product_claim() -> None:
    svc = LLMService()
    text = svc._identity_response_text("are you google gemini?").lower()
    assert "not an official google product" in text
