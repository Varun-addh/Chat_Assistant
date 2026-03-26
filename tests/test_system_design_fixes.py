"""
Comprehensive tests for Phase 9 system design fixes.
Covers: exception sanitization, enum consolidation, auth guards,
rate limiting, input validation, cache module-level, size guards,
dead code removal, filename sanitization.
"""
import re
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from collections import OrderedDict

from fastapi import HTTPException
from fastapi.testclient import TestClient


# ─── Issue #1: generate_architecture must NOT leak str(e) ───

def test_generate_architecture_sanitizes_exception():
    """Ensure internal error details are never sent to the client."""
    from app.routers.diagrams import generate_architecture
    import inspect
    src = inspect.getsource(generate_architecture)
    assert "str(e)" not in src, "generate_architecture still leaks str(e) to client"
    assert "Architecture generation failed. Please try again." in src


# ─── Issue #2: SSE stream must NOT leak str(e) ───

def test_sse_stream_error_sanitized():
    """The architecture SSE generator must NOT embed raw error text."""
    from app.routers import questions
    import inspect
    src = inspect.getsource(questions)
    # The old pattern: f"data: ... {err_msg}" where err_msg = str(e)
    assert "Generation failed:** {err_msg}" not in src, "SSE stream still leaks err_msg"


# ─── Issue #3: _fallback_view must NOT embed raw error text ───

def test_fallback_view_no_error_leak():
    """_fallback_view explanation must not contain raw exception text."""
    from app.routers import questions
    import inspect
    src = inspect.getsource(questions)
    assert '"- Error: " + _safe_ascii(err_msg)' not in src
    assert '"Error: " + _safe_ascii(err_msg)' not in src


# ─── Issue #4: Single source of truth for ArchitectureViewType ───

def test_architecture_view_type_single_source():
    """ArchitectureViewType must be imported from schemas, not redefined."""
    from app.schemas import ArchitectureViewType as SchemaType
    from app.services.architecture.architecture_generator import ArchitectureViewType as GenType
    # Both references should resolve to the same class
    assert SchemaType is GenType, "ArchitectureViewType is still duplicated"


def test_diagram_style_single_source():
    """DiagramStyle must be imported from schemas, not redefined."""
    from app.schemas import DiagramStyle as SchemaStyle
    from app.services.architecture.architecture_generator import DiagramStyle as GenStyle
    assert SchemaStyle is GenStyle, "DiagramStyle is still duplicated"


# ─── Issue #5: render_mermaid has per-IP rate limiting ───

def test_render_mermaid_has_rate_limit_dependency():
    """render_mermaid endpoints must have the rate-limit dependency."""
    from app.routers.diagrams import render_mermaid, render_mermaid_get
    import inspect

    sig_post = inspect.signature(render_mermaid)
    assert "_rl" in sig_post.parameters, "render_mermaid POST missing rate-limit dependency"

    sig_get = inspect.signature(render_mermaid_get)
    assert "_rl" in sig_get.parameters, "render_mermaid GET missing rate-limit dependency"


def test_rate_limit_render_blocks_excess():
    """The per-IP rate limiter should block after exceeding the limit."""
    import asyncio
    from app.routers.diagrams import _rate_limit_render, _render_ip_hits, _RENDER_RATE_LIMIT

    _render_ip_hits.clear()

    mock_request = MagicMock()
    mock_request.client.host = "10.0.0.99"

    async def _fill_and_exceed():
        for _ in range(_RENDER_RATE_LIMIT):
            await _rate_limit_render(mock_request)
        # Next call should raise 429
        with pytest.raises(HTTPException) as exc_info:
            await _rate_limit_render(mock_request)
        assert exc_info.value.status_code == 429

    asyncio.run(_fill_and_exceed())
    _render_ip_hits.clear()


# ─── Issue #7: recommend_views has input validation ───

def test_recommend_views_has_validation():
    """recommend_views must validate system_description length and user_level pattern."""
    from app.routers.diagrams import recommend_views
    import inspect
    sig = inspect.signature(recommend_views)

    sd_param = sig.parameters["system_description"]
    assert sd_param.default is not inspect.Parameter.empty, "system_description must have Query validation"

    ul_param = sig.parameters["user_level"]
    assert ul_param.default is not inspect.Parameter.empty


# ─── Issue #8: export_markdown has auth + sanitized filename ───

def test_export_markdown_has_auth():
    """export_markdown must require API key authentication."""
    from app.routers.diagrams import export_architecture_markdown
    import inspect
    sig = inspect.signature(export_architecture_markdown)
    assert "_api_key" in sig.parameters, "export_markdown missing auth dependency"


def test_export_markdown_sanitizes_filename():
    """Content-Disposition filename must be sanitized against injection."""
    from app.routers.diagrams import export_architecture_markdown
    import inspect
    src = inspect.getsource(export_architecture_markdown)
    assert "re.sub" in src, "export_markdown should sanitize filename with re.sub"


# ─── Issue #9: Module-level cache ───

def test_mermaid_cache_is_module_level():
    """The Mermaid SVG cache must be a module-level OrderedDict."""
    from app.routers import diagrams
    assert hasattr(diagrams, "_mermaid_cache"), "Missing module-level _mermaid_cache"
    assert isinstance(diagrams._mermaid_cache, OrderedDict)
    assert hasattr(diagrams, "_MERMAID_CACHE_MAX"), "Missing _MERMAID_CACHE_MAX constant"
    assert diagrams._MERMAID_CACHE_MAX == 100


def test_no_function_attribute_cache():
    """render_mermaid must not use function-attribute caching pattern."""
    from app.routers.diagrams import render_mermaid
    import inspect
    src = inspect.getsource(render_mermaid)
    assert "render_mermaid._cache" not in src, "Still using function-attribute cache"
    assert "hasattr(render_mermaid" not in src


# ─── Issue #10: available_views has auth ───

def test_available_views_has_auth():
    """available_views must require API key for consistency."""
    from app.routers.diagrams import get_available_views
    import inspect
    sig = inspect.signature(get_available_views)
    assert "_api_key" in sig.parameters, "available_views missing auth"


# ─── Issue #11: _convert_layer_nodes_to_subgraphs size guard ───

def test_convert_layer_nodes_guard_large_input():
    """_convert_layer_nodes_to_subgraphs must return oversized input unchanged."""
    from app.routers.diagrams import _convert_layer_nodes_to_subgraphs
    huge = "flowchart TD\n" + "  A --> B\n" * 5000  # > 20k chars
    assert len(huge) > 20_000
    result = _convert_layer_nodes_to_subgraphs(huge)
    assert result == huge, "Should return oversized input unchanged"


# ─── Issue #12: _add_sequential_step_numbers size guard ───

def test_add_step_numbers_guard_large_input():
    """_add_sequential_step_numbers must return oversized input unchanged."""
    from app.routers.diagrams import _add_sequential_step_numbers
    huge = "flowchart TD\n" + "  A --> B\n" * 5000
    assert len(huge) > 20_000
    result = _add_sequential_step_numbers(huge)
    assert result == huge, "Should return oversized input unchanged"


# ─── Issue #13: Dead code removed ───

def test_no_dead_if_false_block():
    """The disabled 'if False and style == modern' block must be removed."""
    from app.routers.diagrams import render_mermaid
    import inspect
    src = inspect.getsource(render_mermaid)
    assert "if False and" not in src, "Dead 'if False' block still present"


# ─── Issue #14: Dead Pydantic models removed ───

def test_dead_models_removed():
    """ArchitectureView and ArchitecturePackage must not exist in architecture_generator."""
    from app.services.architecture import architecture_generator as mod
    assert not hasattr(mod, "ArchitectureView"), "Dead ArchitectureView model still present"
    assert not hasattr(mod, "ArchitecturePackage"), "Dead ArchitecturePackage model still present"
    # ViewGenerationPrompt should still exist
    assert hasattr(mod, "ViewGenerationPrompt"), "ViewGenerationPrompt was accidentally removed"


# ─── Issue #15: _svg_placeholder indentation ───

def test_svg_placeholder_proper_indentation():
    """_svg_placeholder must use standard 4-space indentation."""
    from app.routers.diagrams import _svg_placeholder
    import inspect
    src = inspect.getsource(_svg_placeholder)
    lines = src.split("\n")
    for line in lines:
        if line and not line.strip() == "":
            leading = len(line) - len(line.lstrip())
            assert leading % 4 == 0 or line.strip().startswith('<') or line.strip().startswith('"'), \
                f"Bad indentation: {line!r}"


def test_svg_placeholder_returns_valid_svg():
    """_svg_placeholder must return well-formed SVG with message."""
    from app.routers.diagrams import _svg_placeholder
    result = _svg_placeholder("Test error")
    assert result.startswith("<svg")
    assert "Test error" in result
    assert result.endswith("</svg>")


def test_svg_placeholder_escapes_html():
    """_svg_placeholder must escape < and > in messages."""
    from app.routers.diagrams import _svg_placeholder
    result = _svg_placeholder("<script>alert('xss')</script>")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


# ─── Issue #16: No inline `import re` in questions.py ───

def test_no_inline_import_re_in_arch_detection():
    """questions.py must not have inline 'import re' — all should be module-level."""
    import inspect
    from app.routers import questions
    src = inspect.getsource(questions)
    lines = src.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "import re":
            # Module-level import has no leading whitespace (or minimal)
            leading = len(line) - len(line.lstrip())
            assert leading == 0, f"Found inline 'import re' at indentation {leading} (source line {i})"


# ─── _svg_placeholder edge cases ───

def test_svg_placeholder_empty_message():
    """_svg_placeholder must handle empty/None message gracefully."""
    from app.routers.diagrams import _svg_placeholder
    result = _svg_placeholder("")
    assert "Mermaid render failed" in result
    result_none = _svg_placeholder(None)
    assert "Mermaid render failed" in result_none


# ─── Rate limiter edge cases ───

def test_rate_limit_cleanup_on_overflow():
    """Rate limiter must clear state when IP count exceeds threshold."""
    import asyncio
    from app.routers.diagrams import _render_ip_hits, _rate_limit_render

    _render_ip_hits.clear()
    now = time.time()
    for i in range(5001):
        _render_ip_hits[f"10.0.{i // 256}.{i % 256}"] = [now]
    assert len(_render_ip_hits) > 5000

    mock_req = MagicMock()
    mock_req.client.host = "192.168.1.1"
    asyncio.run(_rate_limit_render(mock_req))
    assert len(_render_ip_hits) <= 2
    _render_ip_hits.clear()
