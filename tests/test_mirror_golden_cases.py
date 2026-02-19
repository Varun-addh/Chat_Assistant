from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.fast
def test_mirror_golden_cases_confidence_calibration_ranges():
    """Offline benchmark: deterministic confidence calibration stays within expected ranges.

    This test is intentionally LLM-free: it validates only the calibration logic.
    """

    from app.services.chat.llm_service import LLMService
    from app.services.chat.mirror_ontology import MirrorOntology

    svc = LLMService()

    fixture_path = Path(__file__).parent / "fixtures" / "mirror_golden_cases.json"
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))

    for case in cases:
        ontology_raw = case["ontology"]
        ontology = MirrorOntology(
            topic=ontology_raw.get("topic") or "General",
            primitives=tuple(ontology_raw.get("primitives") or []),
            senior_signals=tuple(ontology_raw.get("senior_signals") or []),
            red_flags=tuple(ontology_raw.get("red_flags") or []),
            likely_followups=tuple(ontology_raw.get("likely_followups") or []),
        )

        c, meta = svc._calibrate_mirror_confidence(
            llm_confidence=float(case.get("llm_confidence", 0.5)),
            ontology=ontology,
            question=case["question"],
            user_answer=case["user_answer"],
            strengths=list(case.get("strengths") or []),
            gaps=list(case.get("gaps") or []),
            schema_drift=bool(case.get("schema_drift")),
        )

        mn = float(case["expect_confidence_min"])
        mx = float(case["expect_confidence_max"])
        assert mn <= c <= mx, (
            f"{case.get('name','<unnamed>')} expected {mn}..{mx}, got {c}. "
            f"meta={meta}"
        )
