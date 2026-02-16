"""Regression test: Interview Intelligence must accept user-provided API keys.

Bug fixed:
- ModernInterviewIntelligenceService._generate_questions_with_llm used llm_service.enabled,
  which only reflects server-side keys and incorrectly prevented use of per-request API keys.

We assert that when a user api_key is provided, the code attempts to create a client
(via _ensure_client) even if enabled=False.
"""

from __future__ import annotations

import asyncio

import pytest

from app.services.interview.interview_intelligence_service import (
    ModernInterviewIntelligenceService,
    QuestionGenerationRequest,
    SearchIntent,
)


class _DummyLLM:
    def __init__(self) -> None:
        self._settings = type("S", (), {"require_user_api_key": False, "llm_provider": "groq"})()
        self.called = False

    @property
    def enabled(self) -> bool:
        # Simulate no server keys configured
        return False

    def _ensure_client(self, api_key=None):
        # If we reach here, the fix worked (we didn't early-return due to enabled=False).
        self.called = True
        return None, "groq"


def test_user_api_key_bypasses_enabled_gate():
    svc = ModernInterviewIntelligenceService()

    # Inject dummy LLM service
    dummy = _DummyLLM()
    svc.llm_service = dummy

    req = QuestionGenerationRequest(
        query="diffusion models",
        intent=SearchIntent(
            primary_topic="data-science",
            question_type="technical",
            difficulty_preference=None,
            keywords=["diffusion", "models"],
            requires_code=False,
            target_companies=[],
        ),
        count=2,
        include_solutions=False,
    )

    async def _run():
        return await svc._generate_questions_with_llm(req, api_key="user_supplied_key")

    out = asyncio.run(_run())
    assert dummy.called is True
    assert isinstance(out, list)
