import pytest

from app.services.llm_service import get_llm_service


@pytest.mark.asyncio
async def test_generate_answer_no_key_does_not_reference_provider(monkeypatch):
    """Regression: `generate_answer()` must not reference `provider` before it's assigned.

    This previously crashed with `NameError: name 'provider' is not defined` before
    `_ensure_client()` was even called.
    """

    svc = get_llm_service("groq")

    def _fake_ensure_client(api_key=None):
        return (None, "groq")

    monkeypatch.setattr(svc, "_ensure_client", _fake_ensure_client)

    answer, truncated = await svc.generate_answer("hi", api_key=None)
    assert answer == "hi"
    assert truncated is False
