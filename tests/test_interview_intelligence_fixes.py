"""Comprehensive tests for Interview Intelligence bug fixes.

Covers:
- normalize_to_list module-level extraction
- Formatting utility extraction (imports from new module)
- Dishonest metadata labels
- Fake company data removal
- /curate requires authentication
- SSRF URL validation in GitHub searcher
- Null-byte sentinel replacement
- Gemini safety settings
- Vector DB ID generation (uuid-based)
- Dead stream_search_results removal
- Personalize results no-op
- Self-rating removal from /features
- WebSocket auth requirement
- SerpAPI validation threshold
- API key resolution centralisation
"""

from __future__ import annotations

import re
import types
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings

# ---------------------------------------------------------------------------
# 1. normalize_to_list extracted to module level
# ---------------------------------------------------------------------------

from app.services.interview.interview_intelligence_service import _normalize_to_list


class TestNormalizeToList:
    def test_none_returns_empty(self):
        assert _normalize_to_list(None) == []

    def test_list_passthrough(self):
        assert _normalize_to_list(["a", "b"]) == ["a", "b"]

    def test_string_wraps_in_list(self):
        assert _normalize_to_list("hello") == ["hello"]

    def test_empty_string_returns_empty(self):
        assert _normalize_to_list("  ") == []

    def test_iterable_converted(self):
        assert _normalize_to_list((1, 2)) == [1, 2]

    def test_non_iterable_returns_empty(self):
        assert _normalize_to_list(42, "field") == []


# ---------------------------------------------------------------------------
# 2. Formatting utilities importable from new module
# ---------------------------------------------------------------------------

def test_formatting_imports_from_utility_module():
    from app.utils.intelligence_formatting import (
        auto_format_code_blocks,
        format_coding_answer_for_interview_tab,
        apply_formatting_to_questions,
        clean_history_metadata,
    )
    assert callable(auto_format_code_blocks)
    assert callable(format_coding_answer_for_interview_tab)
    assert callable(apply_formatting_to_questions)
    assert callable(clean_history_metadata)


def test_formatting_still_importable_from_router():
    """Backwards compat: existing code that imports from the router should still work."""
    from app.routers.interview_intelligence import (
        format_coding_answer_for_interview_tab,
        apply_formatting_to_questions,
    )
    assert callable(format_coding_answer_for_interview_tab)
    assert callable(apply_formatting_to_questions)


def test_auto_format_code_blocks_basic():
    from app.utils.intelligence_formatting import auto_format_code_blocks
    # Already formatted should be returned as-is
    text = "```python\nprint('hi')\n```"
    assert auto_format_code_blocks(text) == text
    # Empty
    assert auto_format_code_blocks("") == ""
    assert auto_format_code_blocks(None) is None


def test_clean_history_metadata():
    from app.utils.intelligence_formatting import clean_history_metadata
    meta = {"topic": "python", "avg_credibility": 0.85}
    cleaned = clean_history_metadata(meta)
    assert "avg_credibility" not in cleaned
    assert cleaned["topic"] == "python"


# ---------------------------------------------------------------------------
# 3. Dishonest verified labels fixed
# ---------------------------------------------------------------------------

def test_no_dishonest_verified_labels_in_generate_and_label():
    """_generate_and_label_questions should NOT label LLM output as verified."""
    import inspect
    from app.services.interview.interview_intelligence_service import (
        EnhancedInterviewIntelligenceService,
    )
    source = inspect.getsource(EnhancedInterviewIntelligenceService._generate_and_label_questions)

    # The method should set is_generated=True (not False) for LLM content
    assert '"is_generated": True' in source or "'is_generated': True" in source
    # Should NOT claim LLM content is verified
    assert '"is_verified": True' not in source


def test_no_interview_database_source_labels():
    """Source labels should NOT say 'Interview Database' for LLM-generated content."""
    import inspect
    from app.services.interview import interview_intelligence_service as mod

    source = inspect.getsource(mod)
    assert "Interview Database" not in source, (
        "Dishonest 'Interview Database' source label still present in service"
    )


def test_source_labels_say_ai_generated():
    """_ground_with_web_search should use 'AI Generated' as source label."""
    import inspect
    from app.services.interview.interview_intelligence_service import (
        EnhancedInterviewIntelligenceService,
    )
    source = inspect.getsource(EnhancedInterviewIntelligenceService._ground_with_web_search)
    assert "AI Generated" in source


# ---------------------------------------------------------------------------
# 4. Fake company data removed
# ---------------------------------------------------------------------------

def test_companies_endpoint_no_fake_counts():
    """The /companies endpoint should NOT include fabricated question_count values."""
    from app.routers.interview_intelligence import router as intelligence_router

    app = FastAPI()
    app.include_router(intelligence_router, prefix="/api/intelligence")
    client = TestClient(app)

    resp = client.get("/api/intelligence/companies")
    assert resp.status_code == 200
    body = resp.json()
    companies = body.get("companies", [])
    assert len(companies) > 0
    # No company should have a question_count key
    for company in companies:
        assert "question_count" not in company, (
            f"Company {company['name']} still has fake question_count"
        )


# ---------------------------------------------------------------------------
# 5. /curate requires auth
# ---------------------------------------------------------------------------

def test_curate_requires_authentication():
    """POST /curate should require valid authentication."""
    from app.routers.interview_intelligence import router as intelligence_router

    app = FastAPI()
    app.include_router(intelligence_router, prefix="/api/intelligence")
    client = TestClient(app)

    resp = client.post(
        "/api/intelligence/curate",
        json={
            "question": "What is a linked list?",
            "answer": "A linear data structure",
            "topic": "data-structures",
        },
    )
    # Should get 401 or 403 (not 200 or 500)
    assert resp.status_code in (401, 403), (
        f"Expected 401/403, got {resp.status_code}: {resp.text}"
    )


# ---------------------------------------------------------------------------
# 6. SSRF URL validation in GitHub searcher
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_github_searcher_rejects_non_github_urls():
    """_extract_questions_from_file should reject non-GitHub URLs."""
    from app.services.chat.dynamic_interview_sources import GitHubSearcher

    searcher = GitHubSearcher()
    item = {"html_url": "https://evil.example.com/blob/main/questions.md"}

    import aiohttp
    async with aiohttp.ClientSession() as session:
        from app.services.chat.dynamic_interview_sources import QuestionDomain
        result = await searcher._extract_questions_from_file(
            item, QuestionDomain.GENERAL_TECHNICAL, session
        )
    assert result == []


# ---------------------------------------------------------------------------
# 7. Null-byte sentinels replaced
# ---------------------------------------------------------------------------

def test_fix_json_escapes_no_null_bytes():
    """_fix_json_escapes should not use null bytes as sentinels."""
    from app.services.interview.interview_intelligence_service import (
        ModernInterviewIntelligenceService,
    )

    svc = ModernInterviewIntelligenceService.__new__(ModernInterviewIntelligenceService)
    test_input = r'{"key": "val\\ue", "nested": "line\nnext"}'
    result = svc._fix_json_escapes(test_input)

    assert "\x00" not in result, "Output still contains null bytes"
    # Valid JSON escapes should be preserved
    assert "\\\\" in result or "\\n" in result


# ---------------------------------------------------------------------------
# 8. Gemini safety settings
# ---------------------------------------------------------------------------

def test_no_block_none_in_service():
    """Gemini safety settings should not use BLOCK_NONE."""
    import inspect
    from app.services.interview import interview_intelligence_service as mod

    source = inspect.getsource(mod)
    # BLOCK_NONE should not appear anymore
    assert "BLOCK_NONE" not in source, "BLOCK_NONE still found in service source"


# ---------------------------------------------------------------------------
# 9. Vector DB ID uses UUID
# ---------------------------------------------------------------------------

def test_store_in_vector_db_uses_uuid():
    """_store_in_vector_db should use uuid-based IDs, not scroll(limit=1)."""
    import inspect
    from app.services.interview.interview_intelligence_service import (
        ModernInterviewIntelligenceService,
    )

    source = inspect.getsource(ModernInterviewIntelligenceService._store_in_vector_db)
    assert "uuid" in source.lower(), "Should use uuid for point IDs"
    assert "scroll" not in source, "Should not use scroll to generate IDs"


# ---------------------------------------------------------------------------
# 10. Dead stream_search_results removed
# ---------------------------------------------------------------------------

def test_no_stream_search_results_method():
    """stream_search_results should be removed (was dead/broken code)."""
    from app.services.interview.interview_intelligence_service import (
        UltraProductionInterviewService,
    )
    assert not hasattr(UltraProductionInterviewService, "stream_search_results"), (
        "stream_search_results should be removed"
    )


# ---------------------------------------------------------------------------
# 11. _personalize_results is a clean no-op
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_personalize_results_passthrough():
    """_personalize_results should return input unmodified."""
    from app.services.interview.interview_intelligence_service import (
        UltraProductionInterviewService,
    )

    svc = UltraProductionInterviewService.__new__(UltraProductionInterviewService)
    data = [{"question": "test"}]
    result = await svc._personalize_results(data, "user123")
    assert result is data


# ---------------------------------------------------------------------------
# 12. Self-rating removed from /features
# ---------------------------------------------------------------------------

def test_features_no_self_rating():
    """/features should not include self-rating."""
    from app.routers.interview_intelligence import router as intelligence_router

    app = FastAPI()
    app.include_router(intelligence_router, prefix="/api/intelligence")
    client = TestClient(app)

    # Mock the lazy-loaded services
    mock_svc = MagicMock()
    mock_svc.enable_reranking = True

    with patch(
        "app.routers.interview_intelligence.ultra_production_service", mock_svc
    ), patch(
        "app.routers.interview_intelligence.interview_intelligence_service", MagicMock()
    ), patch(
        "app.routers.interview_intelligence.enhanced_interview_service", MagicMock()
    ):
        resp = client.get("/api/intelligence/features")

    assert resp.status_code == 200
    body = resp.json()
    assert "rating" not in body, "Self-rating should be removed"
    assert "9-10/10" not in str(body)


# ---------------------------------------------------------------------------
# 13. SerpAPI validation uses stronger threshold
# ---------------------------------------------------------------------------

def test_serpapi_threshold_at_least_3():
    """The fuzzy_overlap threshold for SerpAPI validation should be >= 3."""
    import inspect
    from app.services.interview.interview_intelligence_service import (
        EnhancedInterviewIntelligenceService,
    )

    source = inspect.getsource(
        EnhancedInterviewIntelligenceService._ground_with_web_search
    )
    # Look for the threshold comparison
    match = re.search(r"fuzzy_overlap\s*>=\s*(\d+)", source)
    assert match, "Could not find fuzzy_overlap threshold"
    threshold = int(match.group(1))
    assert threshold >= 3, f"Threshold is {threshold}, should be >= 3"


# ---------------------------------------------------------------------------
# 14. Centralized API key resolution helper exists
# ---------------------------------------------------------------------------

def test_resolve_api_key_helper_exists():
    from app.routers.interview_intelligence import _resolve_api_key, _resolve_api_key_simple
    assert callable(_resolve_api_key)
    assert callable(_resolve_api_key_simple)


# ---------------------------------------------------------------------------
# 15. Formatting unescape still works after extraction
# ---------------------------------------------------------------------------

def test_unescape_whitespace_sequences():
    from app.utils.intelligence_formatting import _unescape_common_whitespace_sequences

    raw = "def foo():\\n    return 1\\n"
    result = _unescape_common_whitespace_sequences(raw)
    assert "\n" in result
    assert "\\n" not in result
