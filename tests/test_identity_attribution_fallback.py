"""Regression test: safe deterministic attribution fallback.

When developer attribution isn't configured, ownership/development questions
should use the exact neutral product statement.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.services.llm.identity import identity_response_text


@pytest.mark.fast
def test_identity_fallback_for_who_developed_you_matches_exact_text(monkeypatch) -> None:
    # Ensure developer info is empty to force fallback behavior.
    monkeypatch.setattr(settings, "app_developer_name", "")
    monkeypatch.setattr(
        settings,
        "app_developer_attribution",
        "Stratax AI is an independently developed platform. For official information about its development or ownership, please refer to Stratax AI’s documentation or website.",
    )

    out = identity_response_text(settings, "who develope you!!!?")
    assert (
        out
        == "Stratax AI is an independently developed platform. For official information about its development or ownership, please refer to Stratax AI’s documentation or website."
    )
