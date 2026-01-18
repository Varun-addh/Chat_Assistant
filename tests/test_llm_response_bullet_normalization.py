"""Unit tests for LLM response post-processing.

The UI expects Markdown lists to use hyphen bullets ("- ") and each bullet to be on a
single line. Some providers occasionally emit unicode bullets ("•") or put the bullet
marker on its own line, which breaks markdown rendering.

These tests ensure the backend normalizes that output before returning it.
"""

from __future__ import annotations

import pytest
from app.services.llm_service import LLMService


@pytest.mark.fast
def test_format_response_normalizes_unicode_and_dangling_bullets():
    svc = LLMService()

    raw = (
        "Java and Python are both popular programming languages, but they have different strengths.\n\n"
        "•\n"
        "Ease of learning: Python is generally considered easier.\n"
        "•\n"
        "Performance: Java is often faster and more efficient.\n"
    )

    out = svc._format_response(raw)

    assert "•" not in out
    assert "- Ease of learning:" in out
    assert "- Performance:" in out


@pytest.mark.fast
def test_format_response_does_not_touch_fenced_code_blocks():
    svc = LLMService()

    raw = (
        "Example:\n"
        "```python\n"
        "print('• should stay literal in code')\n"
        "```\n"
        "•\n"
        "This should become a markdown bullet.\n"
    )

    out = svc._format_response(raw)

    assert "```python\nprint('• should stay literal in code')\n```" in out
    assert "- This should become a markdown bullet." in out


@pytest.mark.fast
def test_format_response_converts_colon_label_runs_into_bullets():
    svc = LLMService()

    raw = (
        "Key Features of Java:\n\n"
        "Platform Independence: Runs on any JVM.\n\n"
        "Object-Oriented: Uses classes and objects.\n\n"
        "Simple Syntax: Familiar to C/C++ developers.\n"
    )

    out = svc._format_response(raw)

    assert "- Platform Independence: Runs on any JVM." in out
    assert "- Object-Oriented: Uses classes and objects." in out
    assert "- Simple Syntax: Familiar to C/C++ developers." in out


@pytest.mark.fast
def test_format_response_removes_unbalanced_bold_markers():
    svc = LLMService()

    raw = (
        "This line is broken: Machine learning** include:\n"
        "Another ok line.\n"
    )

    out = svc._format_response(raw)
    assert "**" not in out
