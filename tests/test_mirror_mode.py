"""Mirror mode routing tests.

These tests are deliberately LLM-free by monkeypatching the router's llm_service.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.questions import router as questions_router


@pytest.mark.fast
def test_mirror_requires_user_answer(monkeypatch):
    app = FastAPI()
    app.include_router(questions_router, prefix="/api")
    client = TestClient(app)

    import app.routers.questions as questions_mod

    monkeypatch.setattr(questions_mod, "get_user_id_from_request", lambda _req: "test_mirror_user")

    resp = client.post(
        "/api/question",
        headers={"X-API-Key": "test_key"},
        json={
            "session_id": str(uuid.uuid4()),
            "question": "Explain caching",
            "mode": "mirror",
            "user_answer": "",
            "stream": False,
            "save_to_history": False,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("ui_action") == "collect_mirror_answer"
    assert body.get("answer") == ""


@pytest.mark.fast
def test_mirror_calls_generate_mirror_report_structured(monkeypatch):
    app = FastAPI()
    app.include_router(questions_router, prefix="/api")
    client = TestClient(app)

    import app.routers.questions as questions_mod

    monkeypatch.setattr(questions_mod, "get_user_id_from_request", lambda _req: "test_mirror_user")

    class _DummyLLM:
        def __init__(self) -> None:
            self.called = False
            self.kwargs = None

        def _is_identity_question(self, _q: str) -> bool:
            return False

        def _identity_response_text(self, q: str) -> str:
            return q

        async def generate_mirror_report_structured(self, **kwargs):
            self.called = True
            self.kwargs = kwargs

            md = "## Interview Mirror (Caching)\n\n**Strengths**\n- Mentions TTL\n"
            report = {
                "topic": "Caching",
                "message": "",
                "strengths": ["Mentions TTL"],
                "gaps": [],
                "red_flags": [],
                "likely_followups": [],
                "upgrade_lines": [],
                "confidence": 0.6,
                "_meta": {"parse_ok": True, "validation_ok": True, "schema_drift": False},
            }
            return (md, False, report)

    dummy = _DummyLLM()
    monkeypatch.setattr(questions_mod, "llm_service", dummy)

    resp = client.post(
        "/api/question",
        headers={"X-API-Key": "test_key"},
        json={
            "session_id": str(uuid.uuid4()),
            "question": "Explain caching",
            "mode": "mirror",
            "user_answer": "Caching stores responses to speed up.",
            "stream": False,
            "save_to_history": False,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert dummy.called is True
    assert body["answer"].startswith("## Interview Mirror")
    assert dummy.kwargs["question"] == "Explain caching"
    assert "user_answer" in dummy.kwargs


@pytest.mark.fast
def test_mirror_low_confidence_guard_vague_answer(monkeypatch):
    """Regression: vague answers should not get 'upgrade lines' with false authority."""
    import asyncio

    from app.services.chat.llm_service import LLMService

    svc = LLMService()

    async def _fake_generate_text(prompt: str, **kwargs):
        # Simulate model output: low confidence + too many gaps + upgrade lines.
        return (
            '{'
            '"topic":"Caching",'
            '"message":"",'
            '"strengths":["Mentions caching"],'
            '"gaps":["gap1","gap2","gap3","gap4"],'
            '"red_flags":[], '
            '"likely_followups":[], '
            '"upgrade_lines":["Overconfident line 1","Overconfident line 2"],'
            '"confidence":0.2'
            '}'
        )

    # Avoid real LLM calls for ontology generation and report generation.
    async def _fake_ontology_get(**kwargs):
        from app.services.chat.mirror_ontology import MirrorOntology

        return MirrorOntology(
            topic="General",
            primitives=(),
            senior_signals=(),
            red_flags=(),
            likely_followups=(),
        )

    monkeypatch.setattr(svc, "generate_text", _fake_generate_text)
    monkeypatch.setattr(svc._mirror_ontology, "get", _fake_ontology_get)

    async def _run():
        md, _truncated, report = await svc.generate_mirror_report_structured(
            question="What is caching?",
            user_answer="It makes it faster.",
            depth="standard",
            api_key="test_key",
        )
        return md, report

    md, report = asyncio.run(_run())

    assert report["confidence"] < 0.4
    assert report["upgrade_lines"] == []
    assert len(report["gaps"]) <= 2
    assert "Low confidence" in (report.get("message") or "")
    assert "Low confidence" in md


@pytest.mark.fast
def test_mirror_rewrites_meta_advice_upgrade_lines(monkeypatch):
    """If the model returns coaching meta-advice, we rewrite into speakable lines."""
    import asyncio

    from app.services.chat.llm_service import LLMService

    svc = LLMService()

    call_count = {"n": 0}

    async def _fake_generate_text(prompt: str, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return (
                '{'
                '"topic":"Data Science",'
                '"message":"",'
                '"strengths":["Mentions analysis"],'
                '"gaps":["Does not mention ML/statistics"],'
                '"red_flags":[], '
                '"likely_followups":[], '
                '"upgrade_lines":["Provide a more comprehensive definition that includes multiple disciplines"],'
                '"confidence":0.8'
                '}'
            )
        # Second call: rewrite upgrade lines
        return '{"upgrade_lines":["Data science combines statistics, machine learning, and domain expertise to turn data into decisions."]}'

    async def _fake_ontology_get(**kwargs):
        from app.services.chat.mirror_ontology import MirrorOntology

        return MirrorOntology(
            topic="Data Science",
            primitives=(),
            senior_signals=(),
            red_flags=(),
            likely_followups=(),
        )

    monkeypatch.setattr(svc, "generate_text", _fake_generate_text)
    monkeypatch.setattr(svc._mirror_ontology, "get", _fake_ontology_get)

    async def _run():
        _md, _truncated, report = await svc.generate_mirror_report_structured(
            question="Define data science",
            user_answer="It is about analyzing data.",
            depth="standard",
            api_key="test_key",
        )
        return report

    report = asyncio.run(_run())
    assert call_count["n"] >= 2
    assert report["confidence"] >= 0.4
    assert report["upgrade_lines"]
    assert not report["upgrade_lines"][0].lower().startswith("provide ")

@pytest.mark.fast
def test_mirror_softens_harsh_red_flags(monkeypatch):
    """Regression: harsh red-flag phrasing like 'confuses' should be auto-softened."""
    # NOTE: This test file previously mixed tabs/spaces in this function.
    # Keep indentation consistent to avoid IndentationError.
    import asyncio

    from app.services.chat.llm_service import LLMService
    from app.services.chat.mirror_ontology import MirrorOntology

    svc = LLMService()

    call_count = {"n": 0}

    async def _fake_generate_text(prompt: str, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return (
                "{"
                '"topic":"API",'
                '"message":"",'
                '"strengths":["Understands bridge concept"],'
                '"gaps":["No mention of types (REST/SOAP)"],'
                '"red_flags":["Confuses API role with implementation details"], '
                '"likely_followups":[], '
                '"upgrade_lines":[], '
                '"confidence":0.8'
                "}"
            )
        # Second call: soften red flags
        return '{"red_flags":["May sound junior; narrow framing (backend-frontend only)"]}'

    async def _fake_ontology_get(**kwargs):
        return MirrorOntology(
            topic="API",
            primitives=(),
            senior_signals=(),
            red_flags=(),
            likely_followups=(),
        )

    monkeypatch.setattr(svc, "generate_text", _fake_generate_text)
    monkeypatch.setattr(svc._mirror_ontology, "get", _fake_ontology_get)

    async def _run():
        _md, _truncated, report = await svc.generate_mirror_report_structured(
            question="What is API?",
            user_answer="API is interaction between backend and frontend.",
            depth="standard",
            api_key="test_key",
        )
        return report

    report = asyncio.run(_run())
    assert call_count["n"] >= 2
    assert report["confidence"] >= 0.4
    assert report["red_flags"]
    assert "confuses" not in report["red_flags"][0].lower()
    assert any(
        x in report["red_flags"][0].lower() for x in ["may sound", "narrow framing", "junior"]
    )


@pytest.mark.fast
def test_mirror_parses_prose_wrapped_json(monkeypatch):
    """Models sometimes wrap JSON in prose/fences; extraction should still work."""
    import asyncio

    from app.services.chat.llm_service import LLMService

    svc = LLMService()

    async def _fake_generate_text(prompt: str, **kwargs):
        return (
            "Sure — here is the analysis:\n"
            "```json\n"
            "{"
            "\"topic\":\"Caching\"," 
            "\"message\":\"\"," 
            "\"strengths\":[\"Mentions TTL\"],"
            "\"gaps\":[\"No eviction strategy\"],"
            "\"red_flags\":[],"
            "\"likely_followups\":[],"
            "\"upgrade_lines\":[\"Caching stores computed results to reduce latency and load; TTL + invalidation keeps it correct.\"],"
            "\"confidence\":0.8"
            "}\n"
            "```"
        )

    async def _fake_ontology_get(**kwargs):
        from app.services.chat.mirror_ontology import MirrorOntology
        return MirrorOntology(topic="Caching", primitives=(), senior_signals=(), red_flags=(), likely_followups=())

    monkeypatch.setattr(svc, "generate_text", _fake_generate_text)
    monkeypatch.setattr(svc._mirror_ontology, "get", _fake_ontology_get)

    async def _run():
        _md, _truncated, report = await svc.generate_mirror_report_structured(
            question="What is caching?",
            user_answer="Caching makes things faster.",
            depth="standard",
            api_key="test_key",
        )
        return report

    report = asyncio.run(_run())
    assert report.get("topic") == "Caching"
    assert report.get("confidence") >= 0.4
    assert isinstance((report.get("_meta") or {}).get("parse_ok"), bool)


@pytest.mark.fast
def test_mirror_schema_drift_forces_low_confidence(monkeypatch):
    """If the model output is incomplete, force low-confidence behavior safely."""
    import asyncio

    from app.services.chat.llm_service import LLMService

    svc = LLMService()

    async def _fake_generate_text(prompt: str, **kwargs):
        # Missing most keys => schema drift
        return '{"topic":"Caching","confidence":0.95}'

    async def _fake_ontology_get(**kwargs):
        from app.services.chat.mirror_ontology import MirrorOntology
        return MirrorOntology(topic="Caching", primitives=(), senior_signals=(), red_flags=(), likely_followups=())

    monkeypatch.setattr(svc, "generate_text", _fake_generate_text)
    monkeypatch.setattr(svc._mirror_ontology, "get", _fake_ontology_get)

    async def _run():
        _md, _truncated, report = await svc.generate_mirror_report_structured(
            question="What is caching?",
            user_answer="Caching makes it faster.",
            depth="standard",
            api_key="test_key",
        )
        return report

    report = asyncio.run(_run())
    assert (report.get("_meta") or {}).get("schema_drift") is True
    assert report.get("confidence", 1.0) < 0.4
    assert "Low confidence" in (report.get("message") or "")


@pytest.mark.fast
def test_mirror_policy_has_injection_shield():
    from app.prompts.mirror_policies import MIRROR_MODE

    text = MIRROR_MODE.text.lower()
    assert "untrusted" in text
    assert "ignore any instructions" in text


@pytest.mark.fast
def test_mirror_confidence_calibration_downgrades_overconfident_vague_answer(monkeypatch):
    import asyncio

    from app.services.chat.llm_service import LLMService

    svc = LLMService()

    async def _fake_generate_text(prompt: str, **kwargs):
        # Model claims high confidence, but answer is vague.
        return (
            '{'
            '"topic":"System Design",'
            '"message":"",'
            '"strengths":[], '
            '"gaps":["No scale assumptions","No data model","No trade-offs"],'
            '"red_flags":[], '
            '"likely_followups":[], '
            '"upgrade_lines":["Mention scale and trade-offs"],'
            '"confidence":0.95'
            '}'
        )

    async def _fake_ontology_get(**kwargs):
        from app.services.chat.mirror_ontology import MirrorOntology
        return MirrorOntology(
            topic="System Design",
            primitives=("latency", "throughput", "data model", "cache", "queue"),
            senior_signals=(),
            red_flags=(),
            likely_followups=(),
        )

    monkeypatch.setattr(svc, "generate_text", _fake_generate_text)
    monkeypatch.setattr(svc._mirror_ontology, "get", _fake_ontology_get)

    async def _run():
        _md, _truncated, report = await svc.generate_mirror_report_structured(
            question="Design a URL shortener",
            user_answer="Use microservices and make it scalable.",
            depth="standard",
            api_key="test_key",
        )
        return report

    report = asyncio.run(_run())
    # Calibration should force this into low-confidence territory.
    assert report.get("confidence", 1.0) < 0.4
    assert (report.get("_meta") or {}).get("confidence_calibrated") is True
    assert "Low confidence" in (report.get("message") or "")


@pytest.mark.fast
def test_mirror_confidence_calibration_can_bump_pessimistic_model(monkeypatch):
    import asyncio

    from app.services.chat.llm_service import LLMService

    svc = LLMService()

    async def _fake_generate_text(prompt: str, **kwargs):
        # Model claims low-ish confidence, but the answer covers many primitives.
        return (
            '{'
            '"topic":"Caching",'
            '"message":"",'
            '"strengths":["Mentions TTL","Mentions invalidation"],'
            '"gaps":["No eviction policy detail"],'
            '"red_flags":[], '
            '"likely_followups":[], '
            '"upgrade_lines":[], '
            '"confidence":0.35'
            '}'
        )

    async def _fake_ontology_get(**kwargs):
        from app.services.chat.mirror_ontology import MirrorOntology
        return MirrorOntology(
            topic="Caching",
            primitives=("ttl", "invalidation", "eviction", "cache stampede"),
            senior_signals=(),
            red_flags=(),
            likely_followups=(),
        )

    monkeypatch.setattr(svc, "generate_text", _fake_generate_text)
    monkeypatch.setattr(svc._mirror_ontology, "get", _fake_ontology_get)

    answer = (
        "Caching stores computed results to reduce latency and backend load. "
        "I’d use TTL plus explicit invalidation on writes, and watch for stampedes with request coalescing."
    )

    async def _run():
        _md, _truncated, report = await svc.generate_mirror_report_structured(
            question="What is caching?",
            user_answer=answer,
            depth="standard",
            api_key="test_key",
        )
        return report

    report = asyncio.run(_run())
    assert (report.get("_meta") or {}).get("confidence_calibrated") is True
    # Should not be forced into the low-confidence guard given high coverage + good length.
    assert report.get("confidence", 0.0) >= 0.35