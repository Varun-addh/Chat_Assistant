"""Regression tests for Interview Intelligence formatting.

We have seen upstream answers/code arrive with literal '\\n' sequences instead of
actual newlines, which breaks markdown rendering in the UI.

This test ensures we unescape common whitespace sequences before formatting.
"""

from __future__ import annotations

from app.routers.interview_intelligence import format_coding_answer_for_interview_tab


def test_formatting_unescapes_code_solution_newlines():
    out = format_coding_answer_for_interview_tab(
        answer="Concept: BFS\\nExplanation: use a queue",
        code_solution="def f():\\n\\treturn 1\\n",
        is_coding=True,
        language="python",
        time_complexity=None,
        space_complexity=None,
    )

    # Should not contain literal backslash-n sequences in the rendered markdown
    assert "\\n" not in out
    # Should include a proper fenced code block with actual newlines
    assert "```python" in out
    assert "def f():\n\treturn 1" in out
