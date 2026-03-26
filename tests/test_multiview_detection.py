"""
Proper pytest tests for multi-view architecture detection logic from questions.py
"""
import re
import pytest
from app.config import settings


def _detect_architecture(question: str) -> bool:
    """Replicate the detection logic from questions.py for unit testing."""
    q_lower = question.lower()

    has_explicit = any(kw in q_lower for kw in settings.architecture_detection_explicit_keywords)

    has_design_verb = any(kw in q_lower for kw in settings.architecture_detection_design_verbs)
    looks_like_system_design_phrase = bool(
        re.search(r"\bdesign\s+(a|an|the)?\s*system\b", q_lower)
        or re.search(r"\bdesign\s+(a|an|the)?\s*(platform|service|api|backend|architecture)\b", q_lower)
        or re.search(r"\bbuild\s+(a|an|the)?\s*(platform|service|api|backend)\b", q_lower)
    )

    has_system_concepts = (
        sum([
            any(kw in q_lower for kw in settings.architecture_detection_system_concepts_scale),
            any(kw in q_lower for kw in settings.architecture_detection_system_concepts_data),
            any(kw in q_lower for kw in settings.architecture_detection_system_concepts_infra),
        ])
        >= 2
    )

    is_code_problem = any(kw in q_lower for kw in settings.architecture_detection_code_problem_keywords)

    return (
        has_explicit
        or has_system_concepts
        or (has_design_verb and looks_like_system_design_phrase)
    ) and not is_code_problem


# --- Positive cases: should be detected as system design ---

@pytest.mark.parametrize("question", [
    "Design a system for video streaming like Netflix",
    "System design: Design YouTube",
    "How would you design an architecture for Uber?",
    "Design a high level design for WhatsApp",
    "Create architecture for a distributed cache",
    "Design a platform for food delivery",
    "Build a backend for a social media application",
])
def test_architecture_positive_detection(question):
    assert _detect_architecture(question) is True, f"Should detect as architecture: {question}"


# --- Negative cases: should NOT be detected as system design ---

@pytest.mark.parametrize("question", [
    "Write a function to reverse a string",
    "Implement a sorting algorithm",
    "Design a class for a binary tree",
    "How do I use Python decorators?",
    "Explain the time complexity of quicksort",
])
def test_architecture_negative_detection(question):
    assert _detect_architecture(question) is False, f"Should NOT detect as architecture: {question}"
