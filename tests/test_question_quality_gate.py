"""Tests for the generated-question quality gate.

The Qdrant question cache has no expiry, so anything written is served
indefinitely and outlives the prompt that produced it. These tests pin the gate
that keeps known-bad shapes out, using the real questions that scored 0.0 in
production as the fixtures.
"""

import pytest

from app.utils.question_quality import (
    filter_low_quality,
    is_formulaic,
    is_low_quality,
    rejection_reason,
)


# Verbatim from captured_failures/ — every one of these scored 0.0.
REAL_BAD_QUESTIONS = [
    "Explain how Agentic AI uses HTML for user interfaces",
    "Explain the role of HTML in Agentic AI applications",
    "Explain how JavaScript is used in Agentic AI",
    "Explain how Agentic AI uses Python for decision making",
    "Explain how Agentic AI uses Python for natural language processing",
]

# Questions a real interviewer would ask — these must survive the gate.
GOOD_QUESTIONS = [
    # Regression: rejected in production on 2026-07-27 as formulaic_template.
    # Shares the "role of X in Y" template with the HTML junk below, but X is a
    # concept intrinsic to the topic, not an unrelated named technology.
    "Discuss the role of explainability in Agentic AI systems, especially in regulated domains",
    "Discuss the role of caching in a read-heavy architecture",
    "How do you handle race conditions in Go?",
    "You need to scale a write-heavy service to 50k QPS. Walk me through your approach.",
    "When would you choose a message queue over a direct RPC call?",
    "Explain database indexing and when it hurts more than it helps.",
    "Walk me through what happens under the hood when a goroutine blocks on a channel.",
    "Tell me about a time you had to debug a production outage.",
    "What is a B-tree and why do databases use one?",
]


@pytest.mark.parametrize("question", REAL_BAD_QUESTIONS)
def test_rejects_real_production_failures(question):
    assert is_formulaic(question), f"should be caught as formulaic: {question}"
    assert is_low_quality(question)
    assert rejection_reason(question) == "formulaic_template"


@pytest.mark.parametrize("question", GOOD_QUESTIONS)
def test_keeps_legitimate_questions(question):
    assert not is_low_quality(question), f"false positive on: {question}"
    assert rejection_reason(question) is None


@pytest.mark.parametrize(
    "question,reason",
    [
        ("", "empty"),
        ("   ", "empty"),
        (None, "empty"),
        ("What is X?", "too_short (10 chars)"),
    ],
)
def test_rejects_stub_text(question, reason):
    assert is_low_quality(question)
    assert rejection_reason(question) == reason


def test_filter_splits_kept_and_rejected():
    class Q:
        def __init__(self, question):
            self.question = question

    items = [Q(q) for q in GOOD_QUESTIONS[:2] + REAL_BAD_QUESTIONS[:3]]
    kept, rejected = filter_low_quality(items)

    assert [q.question for q in kept] == GOOD_QUESTIONS[:2]
    assert [q.question for q in rejected] == REAL_BAD_QUESTIONS[:3]


def test_filter_supports_dicts_via_text_of():
    items = [{"question": GOOD_QUESTIONS[0]}, {"question": REAL_BAD_QUESTIONS[0]}]
    kept, rejected = filter_low_quality(items, text_of=lambda d: d["question"])

    assert len(kept) == 1 and len(rejected) == 1
    assert kept[0]["question"] == GOOD_QUESTIONS[0]


def test_filter_handles_empty_input():
    assert filter_low_quality([]) == ([], [])


def test_cleanup_script_shares_the_gate_definition():
    """The eviction script and the write gate must not drift apart.

    If they diverge, --formulaic evicts shapes that the gate still allows to be
    written, so the next search regenerates exactly what was just cleaned.
    """
    import app.utils.question_quality as gate

    source = (
        __import__("pathlib").Path("scripts/clear_question_cache.py").read_text(encoding="utf-8")
    )
    assert "from app.utils.question_quality import is_formulaic" in source
    assert "_FORMULAIC_PATTERNS = [" not in source, "script re-declared its own patterns"
    assert gate.FORMULAIC_PATTERNS, "shared patterns must be non-empty"
