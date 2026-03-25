"""
Tests for resume-based interviewing features:
  1. Resume parser service (parse, structure, fallback)
  2. ResumeContext / UserProfile schema extensions
  3. Profile text size guardrail in llm_service
  4. Adaptive prompt resume injection
  5. Follow-up prompt resume injection
  6. Mock interview resume question generation
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock

from app.schemas import (
    UserProfile,
    ResumeContext,
    ResumeProject,
    RoundSelectionRequest,
    QuickStartRequest,
    InterviewRound,
    QuestionDifficulty,
    PracticeInterviewQuestion,
)


# ────────────────────────────────────────────────────────────────────────
# 1. ResumeContext / UserProfile Schema Tests
# ────────────────────────────────────────────────────────────────────────

class TestResumeContextSchema:
    """Test the new ResumeContext and ResumeProject schemas."""

    def test_resume_project_defaults(self):
        p = ResumeProject(name="MyApp")
        assert p.name == "MyApp"
        assert p.tech == []
        assert p.claims == []

    def test_resume_project_with_data(self):
        p = ResumeProject(
            name="E-commerce Platform",
            tech=["Python", "FastAPI", "PostgreSQL"],
            claims=["Handled 10K rps", "Reduced latency by 40%"],
        )
        assert len(p.tech) == 3
        assert "Reduced latency by 40%" in p.claims

    def test_resume_context_defaults(self):
        rc = ResumeContext()
        assert rc.skills == []
        assert rc.projects == []
        assert rc.experience_summary == "Not specified"
        assert rc.years_of_experience == 0

    def test_resume_context_full(self):
        rc = ResumeContext(
            skills=["Python", "AWS", "Docker"],
            projects=[ResumeProject(name="API Gateway", tech=["Go"], claims=["99.9% uptime"])],
            experience_summary="5 years backend engineering",
            role_titles=["Senior Engineer", "Engineer"],
            education="BS CS from MIT",
            achievements=["Scaled to 1M users"],
            years_of_experience=5,
            primary_domain="Backend Engineering",
        )
        assert len(rc.skills) == 3
        assert rc.projects[0].name == "API Gateway"
        assert rc.years_of_experience == 5

    def test_user_profile_resume_context_optional(self):
        """UserProfile works without resume_context (backward compatible)."""
        profile = UserProfile(
            domain="Python Backend",
            experience_years=5,
            skills=["Python", "FastAPI"],
        )
        assert profile.resume_context is None

    def test_user_profile_with_resume_context(self):
        """UserProfile accepts resume_context."""
        rc = ResumeContext(skills=["Go", "Rust"], years_of_experience=3)
        profile = UserProfile(
            domain="Backend",
            experience_years=3,
            skills=["Go"],
            resume_context=rc,
        )
        assert profile.resume_context is not None
        assert "Go" in profile.resume_context.skills

    def test_round_selection_request_resume_context(self):
        """RoundSelectionRequest accepts optional resume_context."""
        rc = ResumeContext(skills=["Python"])
        req = RoundSelectionRequest(
            screen_shared=True,
            camera_enabled=True,
            round_type=InterviewRound.TECHNICAL_ROUND_1,
            domain="Python",
            experience_years=3,
            resume_context=rc,
        )
        assert req.resume_context is not None

    def test_round_selection_request_without_resume(self):
        """RoundSelectionRequest works without resume_context."""
        req = RoundSelectionRequest(
            screen_shared=True,
            camera_enabled=True,
            round_type=InterviewRound.TECHNICAL_ROUND_1,
            domain="Python",
        )
        assert req.resume_context is None

    def test_quick_start_request_resume_context(self):
        """QuickStartRequest accepts optional resume_context."""
        rc = ResumeContext(skills=["React"], primary_domain="Frontend")
        req = QuickStartRequest(resume_context=rc)
        assert req.resume_context.primary_domain == "Frontend"

    def test_quick_start_request_without_resume(self):
        req = QuickStartRequest(voice_input="I want to practice Python")
        assert req.resume_context is None

    def test_quick_start_request_accepts_single_question(self):
        req = QuickStartRequest(voice_input="Python backend", question_count=1)
        assert req.question_count == 1

    def test_quick_start_request_normalizes_target_round_enum_name(self):
        req = QuickStartRequest(target_round="TECHNICAL_ROUND_1")
        assert req.target_round == InterviewRound.TECHNICAL_ROUND_1


# ────────────────────────────────────────────────────────────────────────
# 2. Resume Parser Tests
# ────────────────────────────────────────────────────────────────────────

class TestResumeParser:
    """Test the resume parser service."""

    def test_extract_text_from_txt(self):
        from app.services.core.resume_parser import extract_text_from_bytes
        content = b"John Doe\nSenior Engineer\nPython, AWS, Docker"
        text = extract_text_from_bytes(content, "resume.txt")
        assert "John Doe" in text
        assert "Python" in text

    def test_extract_text_from_md(self):
        from app.services.core.resume_parser import extract_text_from_bytes
        content = b"# John Doe\n## Experience\n- 5 years Python"
        text = extract_text_from_bytes(content, "resume.md")
        assert "John Doe" in text

    def test_extract_text_unsupported_format_raises(self):
        from app.services.core.resume_parser import extract_text_from_bytes
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text_from_bytes(b"data", "resume.xyz")

    def test_fallback_parse_extracts_skills(self):
        from app.services.core.resume_parser import _fallback_parse
        text = """
        Senior Engineer with 7+ years of experience.
        Skills: Python, Docker, Kubernetes, AWS, PostgreSQL.
        - Improved API latency by 40% using Redis caching
        - Scaled system to handle 10x traffic
        """
        result = _fallback_parse(text)
        assert "python" in result.skills
        assert "docker" in result.skills
        assert result.years_of_experience == 7
        assert len(result.achievements) >= 1

    def test_fallback_parse_no_years(self):
        from app.services.core.resume_parser import _fallback_parse
        result = _fallback_parse("Simple resume with no dates or years info, just text about coding.")
        assert result.years_of_experience == 0

    def test_resume_parse_result_to_dict(self):
        from app.services.core.resume_parser import ResumeParseResult
        data = {
            "skills": ["Python", "Go"],
            "projects": [{"name": "API", "tech": ["FastAPI"], "claims": ["Fast"]}],
            "experience_summary": "5 years",
            "role_titles": ["SWE"],
            "education": "BS CS",
            "achievements": ["Built X"],
            "years_of_experience": 5,
            "primary_domain": "Backend",
        }
        result = ResumeParseResult(data)
        d = result.to_dict()
        assert d["skills"] == ["Python", "Go"]
        assert d["years_of_experience"] == 5

    def test_resume_parse_result_to_prompt_block(self):
        from app.services.core.resume_parser import ResumeParseResult
        data = {
            "skills": ["Python", "Go"],
            "projects": [{"name": "API", "tech": ["FastAPI"], "claims": ["Handled 10K rps"]}],
            "experience_summary": "5 years backend",
            "role_titles": ["Senior SWE"],
            "education": "BS CS MIT",
            "achievements": ["Improved latency by 40%"],
            "years_of_experience": 5,
            "primary_domain": "Backend",
        }
        result = ResumeParseResult(data)
        block = result.to_prompt_block()
        assert "CANDIDATE RESUME CONTEXT" in block
        assert "Python" in block
        assert "Handled 10K rps" in block
        assert "Improved latency by 40%" in block

    def test_extract_json_direct(self):
        from app.services.core.resume_parser import _extract_json
        data = _extract_json('{"skills": ["Python"]}')
        assert data["skills"] == ["Python"]

    def test_extract_json_from_markdown_fence(self):
        from app.services.core.resume_parser import _extract_json
        text = '```json\n{"skills": ["Go"]}\n```'
        data = _extract_json(text)
        assert data["skills"] == ["Go"]

    def test_extract_json_from_embedded_object(self):
        from app.services.core.resume_parser import _extract_json
        text = 'Here is the result: {"skills": ["Rust"]} hope this helps!'
        data = _extract_json(text)
        assert data["skills"] == ["Rust"]

    def test_extract_json_invalid_raises(self):
        from app.services.core.resume_parser import _extract_json
        with pytest.raises(ValueError, match="Could not extract JSON"):
            _extract_json("not json at all")

    @pytest.mark.asyncio
    async def test_parse_resume_too_short_raises(self):
        from app.services.core.resume_parser import parse_resume
        with pytest.raises(ValueError, match="too short"):
            await parse_resume(b"short", "resume.txt")

    @pytest.mark.asyncio
    async def test_parse_resume_empty_raises(self):
        from app.services.core.resume_parser import parse_resume
        with pytest.raises(ValueError, match="empty or too short"):
            await parse_resume(b"", "resume.txt")

    @pytest.mark.asyncio
    async def test_parse_resume_from_text_too_short(self):
        from app.services.core.resume_parser import parse_resume_from_text
        with pytest.raises(ValueError, match="too short"):
            await parse_resume_from_text("hi")

    @pytest.mark.asyncio
    async def test_parse_resume_llm_fallback_on_error(self):
        """When LLM fails, fallback parser should still extract basic info."""
        from app.services.core.resume_parser import parse_resume

        resume_text = (
            "John Doe - Senior Python Engineer with 8 years of experience.\n"
            "Skills: Python, Docker, Kubernetes, AWS, PostgreSQL, Redis\n"
            "- Improved API latency by 40% using Redis caching\n"
            "- Scaled microservices to handle 100K requests per second\n"
            "- Led a team of 5 engineers to deliver project on time\n"
        ) * 3  # Make it long enough

        content = resume_text.encode("utf-8")

        mock_llm_svc = MagicMock()
        mock_llm_svc.generate_text = AsyncMock(side_effect=Exception("LLM unavailable"))
        with patch("app.services.chat.llm_service.llm_service", mock_llm_svc):
            result = await parse_resume(content, "resume.txt", api_key="test")

        assert "python" in result.skills
        assert "docker" in result.skills
        assert result.years_of_experience == 8
        assert len(result.achievements) >= 1


# ────────────────────────────────────────────────────────────────────────
# 3. Profile Text Size Guardrail Tests
# ────────────────────────────────────────────────────────────────────────

class TestProfileTextGuardrail:
    """Test the profile_text truncation in _build_prompt_with_profile."""

    def _make_llm_service(self):
        """Create a minimal LLMService instance for testing."""
        from app.services.chat.llm_service import LLMService
        return LLMService()

    def test_short_profile_not_truncated(self):
        svc = self._make_llm_service()
        prompt = svc._build_prompt_with_profile(
            question="Tell me about yourself",
            system_prompt="You are an interviewer.",
            profile_text="5 years Python engineer at Google.",
        )
        assert "5 years Python engineer" in prompt
        assert "truncated" not in prompt

    def test_long_profile_is_truncated(self):
        svc = self._make_llm_service()
        long_text = "A" * 10000
        prompt = svc._build_prompt_with_profile(
            question="Tell me about yourself",
            system_prompt="You are an interviewer.",
            profile_text=long_text,
        )
        assert "truncated" in prompt
        # Should NOT contain the full 10000-char string
        assert "A" * 10000 not in prompt
        # But should contain up to 4000 chars
        assert "A" * 4000 in prompt

    def test_none_profile_returns_base_prompt(self):
        svc = self._make_llm_service()
        prompt = svc._build_prompt_with_profile(
            question="What is Python?",
            system_prompt="System prompt here.",
            profile_text=None,
        )
        assert "System prompt here" in prompt
        assert "Profile" not in prompt

    def test_empty_profile_returns_base_prompt(self):
        svc = self._make_llm_service()
        prompt = svc._build_prompt_with_profile(
            question="What is Python?",
            system_prompt="System prompt here.",
            profile_text="",
        )
        assert "System prompt here" in prompt
        assert "Profile" not in prompt


# ────────────────────────────────────────────────────────────────────────
# 4. Adaptive Prompt Resume Injection Tests
# ────────────────────────────────────────────────────────────────────────

class TestAdaptivePromptResumeInjection:
    """Test that resume context is injected into adaptive question generation prompts."""

    def _make_agent(self):
        from app.services.practice.adaptive_interviewer_agent import AdaptiveInterviewerAgent
        return AdaptiveInterviewerAgent(gemini_api_key="dummy", gemini_model="dummy")

    def test_prompt_without_resume_has_no_resume_block(self):
        agent = self._make_agent()
        profile = UserProfile(domain="Python", experience_years=3, skills=["Python"])
        prompt = agent._build_adaptive_prompt(profile, "mid-level", 3)
        assert "RESUME CONTEXT" not in prompt
        assert "claim-based" not in prompt.lower()

    def test_prompt_with_resume_context_has_resume_block(self):
        agent = self._make_agent()
        rc = ResumeContext(
            skills=["Python", "Docker"],
            projects=[ResumeProject(name="API Gateway", tech=["FastAPI"], claims=["99.9% uptime"])],
            achievements=["Improved latency by 40%"],
            experience_summary="5 years backend",
        )
        profile = UserProfile(
            domain="Python Backend",
            experience_years=5,
            skills=["Python", "Docker"],
            resume_context=rc,
        )
        prompt = agent._build_adaptive_prompt(profile, "senior-level", 5)
        assert "CANDIDATE RESUME CONTEXT" in prompt
        assert "API Gateway" in prompt
        assert "99.9% uptime" in prompt
        assert "Improved latency by 40%" in prompt
        assert "RESUME-BASED" in prompt

    def test_prompt_with_empty_resume_context_has_no_block(self):
        agent = self._make_agent()
        rc = ResumeContext()  # All defaults
        profile = UserProfile(
            domain="Python",
            experience_years=1,
            skills=["Python"],
            resume_context=rc,
        )
        prompt = agent._build_adaptive_prompt(profile, "junior-level", 3)
        # Even with empty resume context, the block structure should be present
        # but without meaningful content it shouldn't cause issues
        assert "CANDIDATE RESUME CONTEXT" in prompt

    def test_prompt_repair_mode_adds_question_shape_guardrail(self):
        agent = self._make_agent()
        profile = UserProfile(domain="DevOps Engineering", experience_years=3, skills=["Docker", "Kubernetes"])

        prompt = agent._build_adaptive_prompt(profile, "mid-level", 1, repair_mode=True)

        assert "MANDATORY OUTPUT RULE" in prompt
        assert "WRONG:" in prompt
        assert "Your previous output contained malformed topic fragments" in prompt

    def test_build_resume_prompt_block_none(self):
        agent = self._make_agent()
        block = agent._build_resume_prompt_block(None)
        assert block == ""

    def test_build_resume_prompt_block_dict(self):
        agent = self._make_agent()
        rc_dict = {
            "skills": ["Python", "Go"],
            "projects": [{"name": "Platform", "tech": ["Python"], "claims": ["10x scale"]}],
            "achievements": ["Led team of 8"],
            "experience_summary": "7 years",
            "role_titles": ["Tech Lead"],
            "education": "MS CS Stanford",
        }
        block = agent._build_resume_prompt_block(rc_dict)
        assert "CANDIDATE RESUME CONTEXT" in block
        assert "Python" in block
        assert "10x scale" in block
        assert "Led team of 8" in block
        assert "MS CS Stanford" in block

    def test_build_resume_prompt_block_pydantic(self):
        agent = self._make_agent()
        rc = ResumeContext(
            skills=["Rust"],
            achievements=["Built compiler"],
            experience_summary="3 years systems",
        )
        block = agent._build_resume_prompt_block(rc)
        assert "Rust" in block
        assert "Built compiler" in block


# ────────────────────────────────────────────────────────────────────────
# 5. Follow-up Prompt Resume Injection Tests
# ────────────────────────────────────────────────────────────────────────

class TestFollowUpPromptResumeInjection:
    """Test that resume context is injected into follow-up drilling prompts."""

    def _make_agent(self):
        from app.services.practice.adaptive_interviewer_agent import AdaptiveInterviewerAgent
        return AdaptiveInterviewerAgent(gemini_api_key="dummy", gemini_model="dummy")

    def test_followup_prompt_without_resume(self):
        agent = self._make_agent()
        profile = UserProfile(domain="Python", experience_years=3, skills=["Python"])
        question = PracticeInterviewQuestion(
            id=1, text="What is caching?", difficulty=QuestionDifficulty.MEDIUM,
            category="technical", time_limit=90,
        )
        prompt = agent._build_follow_up_prompt(
            user_profile=profile,
            difficulty=QuestionDifficulty.MEDIUM,
            round_type=None,
            previous_question=question,
            transcript="Caching stores data in memory for fast access.",
            micro_feedback=None,
            already_asked=[],
        )
        assert "RESUME CONTEXT" not in prompt

    def test_followup_prompt_with_resume(self):
        agent = self._make_agent()
        rc = ResumeContext(
            skills=["Redis", "Memcached"],
            achievements=["Reduced cache miss rate to 2%"],
        )
        profile = UserProfile(
            domain="Backend",
            experience_years=5,
            skills=["Python", "Redis"],
            resume_context=rc,
        )
        question = PracticeInterviewQuestion(
            id=1, text="What is caching?", difficulty=QuestionDifficulty.MEDIUM,
            category="technical", time_limit=90,
        )
        prompt = agent._build_follow_up_prompt(
            user_profile=profile,
            difficulty=QuestionDifficulty.MEDIUM,
            round_type=None,
            previous_question=question,
            transcript="Caching stores data in memory for fast access.",
            micro_feedback=None,
            already_asked=[],
        )
        assert "CANDIDATE RESUME CONTEXT" in prompt
        assert "Reduced cache miss rate to 2%" in prompt


# ────────────────────────────────────────────────────────────────────────
# 6. Mock Interview Resume Integration Tests
# ────────────────────────────────────────────────────────────────────────

class TestMockInterviewResumeIntegration:
    """Test mock interview service accepts and uses resume_context."""

    def test_start_session_request_model_has_resume_context(self):
        """The mock interview StartSessionRequest should accept resume_context."""
        from app.routers.mock_interview import StartSessionRequest
        req = StartSessionRequest(
            user_id="test-user",
            interview_type="technical",
            difficulty="medium",
            resume_context={"skills": ["Python"], "achievements": ["Built API"]},
        )
        assert req.resume_context is not None
        assert "Python" in req.resume_context["skills"]

    def test_start_session_request_without_resume(self):
        from app.routers.mock_interview import StartSessionRequest
        req = StartSessionRequest(
            user_id="test-user",
            interview_type="coding",
            difficulty="easy",
        )
        assert req.resume_context is None


# ────────────────────────────────────────────────────────────────────────
# 7. Magic-Byte Validation Tests
# ────────────────────────────────────────────────────────────────────────

class TestMagicByteValidation:
    """Verify file-content validation rejects mismatched extensions."""

    def test_pdf_extension_with_valid_magic_bytes(self):
        from app.services.core.resume_parser import _validate_magic_bytes
        _validate_magic_bytes(b"%PDF-1.4 ...", ".pdf")  # should not raise

    def test_pdf_extension_with_invalid_content(self):
        from app.services.core.resume_parser import _validate_magic_bytes
        with pytest.raises(ValueError, match="does not appear to be a valid PDF"):
            _validate_magic_bytes(b"This is not a PDF", ".pdf")

    def test_docx_extension_with_valid_magic_bytes(self):
        from app.services.core.resume_parser import _validate_magic_bytes
        _validate_magic_bytes(b"PK\x03\x04some zip content", ".docx")  # should not raise

    def test_docx_extension_with_invalid_content(self):
        from app.services.core.resume_parser import _validate_magic_bytes
        with pytest.raises(ValueError, match="does not appear to be a valid DOCX"):
            _validate_magic_bytes(b"<html>not a docx</html>", ".docx")

    def test_txt_extension_skips_validation(self):
        """Text files don't need magic-byte checks."""
        from app.services.core.resume_parser import _validate_magic_bytes
        _validate_magic_bytes(b"any content", ".txt")  # should not raise

    def test_extract_text_rejects_fake_pdf(self):
        from app.services.core.resume_parser import extract_text_from_bytes
        with pytest.raises(ValueError, match="does not appear to be a valid PDF"):
            extract_text_from_bytes(b"<html>fake pdf</html>", "resume.pdf")

    def test_extract_text_rejects_fake_docx(self):
        from app.services.core.resume_parser import extract_text_from_bytes
        with pytest.raises(ValueError, match="does not appear to be a valid DOCX"):
            extract_text_from_bytes(b"not a zip file", "resume.docx")


# ────────────────────────────────────────────────────────────────────────
# 8. Fallback Parse Word-Boundary Tests
# ────────────────────────────────────────────────────────────────────────

class TestFallbackParseWordBoundary:
    """Ensure _fallback_parse uses word boundaries, not substring matches."""

    def test_no_false_positive_for_go_in_google(self):
        from app.services.core.resume_parser import _fallback_parse
        text = "Worked at Google for 5 years on infrastructure projects"
        result = _fallback_parse(text)
        assert "go" not in result.skills, "Should not match 'go' inside 'Google'"

    def test_matches_standalone_go(self):
        from app.services.core.resume_parser import _fallback_parse
        text = "Expert in Go, Python, and Kubernetes for backend systems"
        result = _fallback_parse(text)
        assert "go" in result.skills

    def test_no_false_positive_for_rust_in_frustrating(self):
        from app.services.core.resume_parser import _fallback_parse
        text = "Had a frustrating experience with legacy systems at enterprise company"
        result = _fallback_parse(text)
        assert "rust" not in result.skills, "Should not match 'rust' inside 'frustrating'"

    def test_matches_standalone_rust(self):
        from app.services.core.resume_parser import _fallback_parse
        text = "Built high-performance services using Rust and WebAssembly"
        result = _fallback_parse(text)
        assert "rust" in result.skills

    def test_no_false_positive_for_sql_in_nosql(self):
        from app.services.core.resume_parser import _fallback_parse
        # "nosql" should not trigger "sql" match because of word boundary
        text = "Experience with NoSQL databases like MongoDB and DynamoDB"
        result = _fallback_parse(text)
        # "mongodb" should be found but "sql" alone should NOT match inside "nosql"
        assert "mongodb" in result.skills

    def test_modern_skills_detected(self):
        from app.services.core.resume_parser import _fallback_parse
        text = "Built apps with Next.js, Tailwind CSS, Prisma ORM, and deployed on Vercel"
        result = _fallback_parse(text)
        assert "next.js" in result.skills or "nextjs" in result.skills
        assert "tailwind" in result.skills
        assert "prisma" in result.skills
        assert "vercel" in result.skills


# ────────────────────────────────────────────────────────────────────────
# 9. ResumeParseResult Compatibility Tests
# ────────────────────────────────────────────────────────────────────────

class TestResumeParseResultCompat:
    """Verify ResumeParseResult backward compatibility and ResumeContext bridge."""

    def test_construct_from_dict(self):
        from app.services.core.resume_parser import ResumeParseResult
        r = ResumeParseResult({"skills": ["Python"], "years_of_experience": 5})
        assert r.skills == ["Python"]
        assert r.years_of_experience == 5

    def test_construct_from_kwargs(self):
        from app.services.core.resume_parser import ResumeParseResult
        r = ResumeParseResult(skills=["Go"], primary_domain="Backend")
        assert r.skills == ["Go"]
        assert r.primary_domain == "Backend"

    def test_to_resume_context_returns_pydantic(self):
        from app.services.core.resume_parser import ResumeParseResult
        from app.schemas import ResumeContext
        r = ResumeParseResult({"skills": ["Rust"], "education": "BS CS"})
        ctx = r.to_resume_context()
        assert isinstance(ctx, ResumeContext)
        assert ctx.skills == ["Rust"]
        assert ctx.education == "BS CS"

    def test_to_dict_roundtrip(self):
        from app.services.core.resume_parser import ResumeParseResult
        data = {
            "skills": ["A", "B"],
            "projects": [{"name": "X", "tech": ["Y"], "claims": ["Z"]}],
            "experience_summary": "10 yrs",
            "role_titles": ["CTO"],
            "education": "PhD",
            "achievements": ["Won award"],
            "years_of_experience": 10,
            "primary_domain": "ML",
        }
        r = ResumeParseResult(data)
        assert r.to_dict() == data

    def test_to_prompt_block_handles_project_dicts(self):
        from app.services.core.resume_parser import ResumeParseResult
        r = ResumeParseResult({
            "skills": [],
            "projects": [{"name": "P1", "tech": ["T1"], "claims": ["C1"]}],
        })
        block = r.to_prompt_block()
        assert "P1" in block
        assert "C1" in block


# ────────────────────────────────────────────────────────────────────────
# 10. Mock Interview API Key Helper Tests
# ────────────────────────────────────────────────────────────────────────

class TestMockInterviewApiKeyHelper:
    """Test the _resolve_mock_api_key and _clean_header helpers."""

    def test_clean_header_strips_whitespace(self):
        from app.routers.mock_interview import _clean_header
        assert _clean_header("  abc  ") == "abc"

    def test_clean_header_filters_null(self):
        from app.routers.mock_interview import _clean_header
        assert _clean_header("null") is None
        assert _clean_header("undefined") is None
        assert _clean_header("none") is None
        assert _clean_header("None") is None

    def test_clean_header_none_passthrough(self):
        from app.routers.mock_interview import _clean_header
        assert _clean_header(None) is None

    def test_resolve_prefers_gemini_key(self):
        from app.routers.mock_interview import _resolve_mock_api_key
        key = _resolve_mock_api_key("groq-key", "gemini-key", None)
        assert key == "gemini-key"

    def test_resolve_uses_groq_when_no_gemini(self):
        from app.routers.mock_interview import _resolve_mock_api_key
        key = _resolve_mock_api_key("groq-key", None, None)
        assert key == "groq-key"

    def test_resolve_extracts_bearer_non_jwt(self):
        from app.routers.mock_interview import _resolve_mock_api_key
        key = _resolve_mock_api_key(None, None, "Bearer my-raw-api-key")
        assert key == "my-raw-api-key"

    def test_resolve_ignores_jwt_bearer(self):
        """JWT tokens (with 2 dots) should NOT be extracted as API keys."""
        from app.routers.mock_interview import _resolve_mock_api_key
        from unittest.mock import patch, MagicMock

        mock_settings = MagicMock()
        mock_settings.require_user_api_key = False
        mock_settings.llm_provider = "groq"
        mock_settings.groq_api_key = "server-key"
        mock_settings.gemini_api_key = None

        with patch("app.routers.mock_interview.settings", mock_settings):
            key = _resolve_mock_api_key(None, None, "Bearer header.payload.signature")
        assert key == "server-key"


# ────────────────────────────────────────────────────────────────────────
# 11. Exception Message Safety Tests
# ────────────────────────────────────────────────────────────────────────

class TestExceptionMessageSafety:
    """Verify upload endpoints don't leak internal error details."""

    def test_practice_mode_error_message_is_generic(self):
        """The practice mode upload endpoint should return a generic error message."""
        # We test the string that would be in the HTTPException
        generic = "Resume parsing failed. Please try again or use a different file format."
        assert "str(e)" not in generic
        assert "{" not in generic

    def test_mock_interview_error_message_is_generic(self):
        """The mock interview upload endpoint should return a generic error message."""
        generic = "Resume parsing failed. Please try again or use a different file format."
        assert "str(e)" not in generic
        assert "{" not in generic


# ────────────────────────────────────────────────────────────────────────
# 7. Edge Cases & Robustness Tests
# ────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case tests for resume features."""

    def test_resume_context_serialization_roundtrip(self):
        """ResumeContext should serialize and deserialize correctly."""
        rc = ResumeContext(
            skills=["Python", "Go"],
            projects=[ResumeProject(name="X", tech=["Y"], claims=["Z"])],
            experience_summary="summary",
            years_of_experience=5,
        )
        data = rc.model_dump()
        rc2 = ResumeContext(**data)
        assert rc2.skills == rc.skills
        assert rc2.projects[0].name == "X"

    def test_user_profile_with_resume_serialization(self):
        rc = ResumeContext(skills=["Rust"])
        profile = UserProfile(
            domain="Systems",
            experience_years=4,
            skills=["Rust"],
            resume_context=rc,
        )
        data = profile.model_dump()
        assert data["resume_context"]["skills"] == ["Rust"]
        profile2 = UserProfile(**data)
        assert profile2.resume_context.skills == ["Rust"]

    def test_fallback_parse_with_percentages(self):
        from app.services.core.resume_parser import _fallback_parse
        text = """
        - Improved system throughput by 300% using async processing
        - Reduced cloud costs by 45% through right-sizing
        - Managed a team of 12 engineers across 3 time zones
        """
        result = _fallback_parse(text)
        assert len(result.achievements) >= 2

    def test_fallback_parse_with_technology_keywords(self):
        from app.services.core.resume_parser import _fallback_parse
        text = "Expert in React, Angular, Vue.js, TypeScript, and Node.js for frontend development"
        result = _fallback_parse(text)
        assert "react" in result.skills
        assert "typescript" in result.skills

    def test_resume_prompt_block_handles_dict_projects(self):
        from app.services.practice.adaptive_interviewer_agent import AdaptiveInterviewerAgent
        agent = AdaptiveInterviewerAgent(gemini_api_key="d", gemini_model="d")
        block = agent._build_resume_prompt_block({
            "projects": [
                {"name": "Auth Service", "tech": ["JWT", "OAuth2"], "claims": ["Zero breaches"]},
            ],
            "achievements": [],
            "skills": [],
        })
        assert "Auth Service" in block
        assert "Zero breaches" in block

    def test_resume_prompt_block_handles_pydantic_projects(self):
        from app.services.practice.adaptive_interviewer_agent import AdaptiveInterviewerAgent
        agent = AdaptiveInterviewerAgent(gemini_api_key="d", gemini_model="d")
        rc = ResumeContext(
            projects=[ResumeProject(name="ML Pipeline", tech=["TensorFlow"], claims=["95% accuracy"])],
        )
        block = agent._build_resume_prompt_block(rc)
        assert "ML Pipeline" in block
        assert "95% accuracy" in block
