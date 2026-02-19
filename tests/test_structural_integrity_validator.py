"""Tests for the structural integrity validator — the final assertion layer."""
import pytest
from app.services.chat.llm_service import LLMService


@pytest.fixture
def svc():
    return LLMService()


# ── 1. Balanced code fences ──────────────────────────────────────────────

class TestBalancedCodeFences:
    def test_already_balanced_fences_untouched(self, svc):
        text = "Some text\n```python\nprint('hi')\n```\nMore text"
        out = svc._structural_integrity_check(text)
        assert out.count("```") == 2

    def test_unclosed_fence_gets_closed(self, svc):
        text = "Some text\n```python\nprint('hi')\nMore code"
        out = svc._structural_integrity_check(text)
        # Must have even fence count now
        assert out.count("```") % 2 == 0
        assert out.rstrip().endswith("```")

    def test_multiple_balanced_fences_untouched(self, svc):
        text = "```js\nfoo()\n```\ntext\n```py\nbar()\n```"
        out = svc._structural_integrity_check(text)
        assert out.count("```") == 4

    def test_three_fences_gets_fourth(self, svc):
        text = "```js\nfoo()\n```\ntext\n```py\nbar()"
        out = svc._structural_integrity_check(text)
        assert out.count("```") % 2 == 0

    def test_no_fences_passthrough(self, svc):
        text = "Just plain text with no code."
        out = svc._structural_integrity_check(text)
        assert "```" not in out


# ── 2. Balanced emphasis markers ─────────────────────────────────────────

class TestBalancedEmphasis:
    def test_balanced_bold_untouched(self, svc):
        text = "This is **bold** text."
        out = svc._structural_integrity_check(text)
        assert "**bold**" in out

    def test_orphan_bold_marker_removed(self, svc):
        text = "This is broken bold** text."
        out = svc._structural_integrity_check(text)
        assert out.count("**") % 2 == 0

    def test_emphasis_inside_code_untouched(self, svc):
        text = "```\nbroken ** marker\n```"
        out = svc._structural_integrity_check(text)
        # Inside code block — should NOT be touched
        assert "broken ** marker" in out

    def test_balanced_italic_untouched(self, svc):
        text = "This is *italic* word."
        out = svc._structural_integrity_check(text)
        assert "*italic*" in out


# ── 3. Mermaid block hygiene ─────────────────────────────────────────────

class TestMermaidHygiene:
    def test_valid_mermaid_block_preserved(self, svc):
        text = "```mermaid\nflowchart TD\n  A-->B\n```"
        out = svc._structural_integrity_check(text)
        assert "flowchart TD" in out
        assert "A-->B" in out

    def test_empty_mermaid_block_removed(self, svc):
        text = "Before\n```mermaid\n```\nAfter"
        out = svc._structural_integrity_check(text)
        assert "```mermaid" not in out
        assert "empty diagram removed" in out

    def test_mermaid_without_header_removed(self, svc):
        text = "```mermaid\nA --> B\nB --> C\n```"
        out = svc._structural_integrity_check(text)
        assert "```mermaid" not in out
        assert "invalid diagram removed" in out

    def test_sequence_diagram_preserved(self, svc):
        text = "```mermaid\nsequenceDiagram\n  Alice->>Bob: Hello\n```"
        out = svc._structural_integrity_check(text)
        assert "sequenceDiagram" in out

    def test_er_diagram_preserved(self, svc):
        text = "```mermaid\nerDiagram\n  USER ||--o{ ORDER : places\n```"
        out = svc._structural_integrity_check(text)
        assert "erDiagram" in out

    def test_no_mermaid_passthrough(self, svc):
        text = "Regular text with ```python\ncode\n```"
        out = svc._structural_integrity_check(text)
        assert "python" in out


# ── 4. Response size limit ───────────────────────────────────────────────

class TestSizeLimit:
    def test_short_response_untouched(self, svc):
        text = "Short response."
        out = svc._structural_integrity_check(text)
        assert out == text

    def test_oversized_response_truncated(self, svc):
        # Build a 40KB response
        text = ("This is a sentence. " * 100 + "\n\n") * 25  # ~50KB
        assert len(text) > 32_000
        out = svc._structural_integrity_check(text)
        assert len(out) <= 35_000  # allow small overhead for truncation message
        assert "response truncated" in out

    def test_truncation_closes_open_fence(self, svc):
        # Build response with unclosed code block that exceeds limit
        preamble = "Some text.\n\n" * 500  # big preamble
        text = preamble + "```python\n" + "x = 1\n" * 5000
        assert len(text) > 32_000
        out = svc._structural_integrity_check(text)
        # Fences should be balanced even after truncation
        assert out.count("```") % 2 == 0

    def test_truncation_at_sentence_boundary(self, svc):
        # Build text that's just over the limit
        base = "This is sentence one. This is sentence two.\n\n"
        text = base * 800  # well over 32K
        out = svc._structural_integrity_check(text)
        # Should end at a clean sentence boundary
        assert "response truncated" in out
        # Should not be cut mid-word
        content_before_marker = out.split("*... (response truncated")[0].rstrip()
        assert content_before_marker.endswith((".",".\n","\n"))  or content_before_marker[-1] in ".!?\n"


# ── Integration: full _format_response pipeline ─────────────────────────

class TestIntegrationWithPipeline:
    def test_format_response_closes_unclosed_fence(self, svc):
        raw = "Here is code:\n```python\ndef foo():\n    return 42"
        out = svc._format_response(raw)
        assert out.count("```") % 2 == 0

    def test_format_response_removes_empty_mermaid(self, svc):
        raw = "Diagram:\n```mermaid\n```\nDone."
        out = svc._format_response(raw)
        assert "```mermaid" not in out

    def test_format_response_preserves_valid_content(self, svc):
        raw = (
            "## Explanation\n\n"
            "Here is how it works:\n\n"
            "- Step 1: Do this\n"
            "- Step 2: Do that\n\n"
            "```python\ndef main():\n    pass\n```\n"
        )
        out = svc._format_response(raw)
        assert "Step 1" in out
        assert "Step 2" in out
        assert "def main():" in out
        assert out.count("```") % 2 == 0
