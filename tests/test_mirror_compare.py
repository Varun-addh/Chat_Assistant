import pytest

from app.services.mirror_compare import (
	compute_mirror_progress,
	find_previous_mirror_attempt,
	format_mirror_progress_markdown,
)


@pytest.mark.fast
def test_compute_mirror_progress_basic():
	prev = {
		"confidence": 0.40,
		"gaps": ["Add constraints", "Discuss trade-offs"],
		"strengths": ["Clear structure"],
		"red_flags": ["Hand-wavy"],
	}
	curr = {
		"confidence": 0.62,
		"gaps": ["Discuss trade-offs", "Mention monitoring"],
		"strengths": ["Clear structure", "Concrete numbers"],
		"red_flags": [],
	}

	p = compute_mirror_progress(prev, curr)
	assert p.confidence_prev == pytest.approx(0.40)
	assert p.confidence_curr == pytest.approx(0.62)
	assert p.confidence_delta == pytest.approx(0.22)
	assert "Add constraints" in p.gaps_closed
	assert "Mention monitoring" in p.new_gaps
	assert "Concrete numbers" in p.new_strengths
	assert "Hand-wavy" in p.red_flags_resolved

	md = format_mirror_progress_markdown(p)
	assert "Progress since your last draft" in md
	assert "Confidence:" in md


@pytest.mark.fast
def test_find_previous_mirror_attempt_matches_latest_for_question():
	history = [
		{"question": "Q1", "report": {"confidence": 0.2}},
		{"question": "Q2", "report": {"confidence": 0.3}},
		{"question": "Q1", "report": {"confidence": 0.4}},
	]
	it = find_previous_mirror_attempt(question="Q1", mirror_history=history)
	assert isinstance(it, dict)
	assert (it.get("report") or {}).get("confidence") == 0.4
