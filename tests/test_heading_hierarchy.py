"""Regression tests for consistent heading hierarchy in copilot answers.

Bug: peer sections rendered at two different heading levels within a single
answer. From a real response to "explain python decorators with a code
example":

    ## **Introduction to Python Decorators**   -> <h2>
    ## **Example Code**                        -> <h2>
    ```python … ```
    **Explanation:**                           -> inline <strong>
    ## **Output**                              -> <h2>
    **Key Points**                             -> inline <strong>

_format_headings_bold only promoted a bold-only line to a heading when it was
preceded by a blank line. Models routinely emit a section label immediately
after a closing code fence with no blank line, so those labels stayed inline
while their peers became headings — hierarchy decided by incidental whitespace.
"""

import pytest

from app.services.chat.llm_service import LLMService


def _format(text: str) -> str:
    # _format_headings_bold is attached to LLMService by
    # response_postprocess.attach_text_postprocess_methods.
    return LLMService._format_headings_bold(None, text)


def test_bold_label_after_code_fence_becomes_heading():
    """The exact production case."""
    text = "## **Example Code**\n```python\nprint(1)\n```\n**Explanation:**\n- it prints"
    out = _format(text)
    assert "## **Explanation**" in out, out


def test_peer_sections_end_up_at_the_same_level():
    """No answer should mix <h2> and inline bold for peer sections."""
    text = (
        "## **Introduction**\n"
        "- intro\n"
        "\n"
        "## **Example Code**\n"
        "```python\n"
        "def f(): pass\n"
        "```\n"
        "**Explanation:**\n"
        "- explanation\n"
        "\n"
        "## **Output**\n"
        "```\n"
        "ok\n"
        "```\n"
        "**Key Points**\n"
        "- point\n"
    )
    out = _format(text)
    for section in ("Introduction", "Example Code", "Explanation", "Output", "Key Points"):
        assert f"## **{section}**" in out, f"{section!r} not promoted:\n{out}"
    # No bold-only section label should survive at the start of a line.
    leftovers = [
        line for line in out.split("\n")
        if line.strip().startswith("**") and line.strip().endswith("**")
        and not line.strip().startswith("##")
    ]
    assert leftovers == [], f"unpromoted section labels: {leftovers}"


def test_blank_line_boundary_still_works():
    text = "Some prose.\n\n**Key Points**\n- a"
    assert "## **Key Points**" in _format(text)


def test_document_start_boundary_still_works():
    assert "## **Overview**" in _format("**Overview**\n- a")


def test_bold_mid_paragraph_is_not_promoted():
    """Inline emphasis inside prose must stay inline."""
    text = "This is **important** context.\nAnd **more** here."
    out = _format(text)
    assert "##" not in out, out


def test_bold_line_directly_after_prose_is_not_promoted():
    """Without a boundary it is emphasis, not a section title."""
    text = "Some prose leading in.\n**Not A Heading**\n- a"
    out = _format(text)
    assert "## **Not A Heading**" not in out, out


def test_bold_inside_code_fence_is_untouched():
    text = "```python\n**not_a_heading** = 1\n```"
    out = _format(text)
    assert "## " not in out
    assert "**not_a_heading** = 1" in out


def test_existing_headings_are_not_double_bolded():
    out = _format("## **Already Bold**")
    assert out.count("**") == 2, out
    assert "## ****" not in out


@pytest.mark.parametrize(
    "label",
    ["**Explanation:**", "**Key Points**", "**Output:**", "**Time Complexity**"],
)
def test_common_section_labels_after_fence(label):
    text = f"```python\nx = 1\n```\n{label}\n- detail"
    out = _format(text)
    expected = label.strip("*").rstrip(":")
    assert f"## **{expected}**" in out, out
