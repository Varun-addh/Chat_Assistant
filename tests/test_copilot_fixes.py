"""Tests for the AI Copilot fixes (assessment issues #3-#20).

Covers: quota enforcement, context heuristics, ambiguity detection,
classify_is_technical heuristic, topic-aware recommendations,
user_tier threading, lighter prompts, identity word-boundary,
merged policies, and PromptFlags intent.
"""
import re
import pytest
from unittest.mock import MagicMock, patch

from app.services.chat.llm_service import LLMService
from app.prompts.builder import PromptFlags, build_default_system_prompt
from app.prompts.policies import COPILOT_INTERVIEW_MODE, INTERVIEW_COACH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_svc() -> LLMService:
    """Create a minimal LLMService for unit testing (no API keys needed)."""
    return LLMService()


# ---------------------------------------------------------------------------
# Fix #9: Merged policies — INTERVIEW_COACH is now an alias
# ---------------------------------------------------------------------------

class TestMergedPolicies:
    def test_interview_coach_is_alias(self):
        """INTERVIEW_COACH should be the same object as COPILOT_INTERVIEW_MODE."""
        assert INTERVIEW_COACH is COPILOT_INTERVIEW_MODE

    def test_merged_policy_has_seniority_calibration(self):
        """The merged policy should contain seniority calibration from the old INTERVIEW_COACH."""
        assert "SENIORITY CALIBRATION" in COPILOT_INTERVIEW_MODE.text

    def test_merged_policy_has_language_variation(self):
        """The merged policy should contain language variation."""
        assert "LANGUAGE VARIATION" in COPILOT_INTERVIEW_MODE.text

    def test_merged_policy_has_answer_first(self):
        """The merged policy should contain ANSWER FIRST."""
        assert "ANSWER FIRST" in COPILOT_INTERVIEW_MODE.text

    def test_no_duplicate_policy_name_in_builder(self):
        """build_default_system_prompt should not include duplicate 'interview_coach' module."""
        prompt = build_default_system_prompt(
            app_name="TestApp",
            developer_name="Dev",
            attribution="attr",
            flags=PromptFlags(),
        )
        # The merged policy name is "copilot_interview_mode"; there's no separate
        # "interview_coach" section.
        assert prompt.count("copilot_interview_mode") <= 1


# ---------------------------------------------------------------------------
# Fix #10: Lighter prompts for greetings/off-topic
# ---------------------------------------------------------------------------

class TestLighterPrompts:
    def test_greeting_prompt_is_shorter(self):
        full = build_default_system_prompt(
            app_name="X", developer_name="D", attribution="A",
            flags=PromptFlags(),
        )
        light = build_default_system_prompt(
            app_name="X", developer_name="D", attribution="A",
            flags=PromptFlags(intent="greeting"),
        )
        assert len(light) < len(full), "Greeting prompt should be shorter than full prompt"

    def test_off_topic_prompt_is_shorter(self):
        full = build_default_system_prompt(
            app_name="X", developer_name="D", attribution="A",
            flags=PromptFlags(),
        )
        light = build_default_system_prompt(
            app_name="X", developer_name="D", attribution="A",
            flags=PromptFlags(intent="off_topic"),
        )
        assert len(light) < len(full)

    def test_greeting_still_has_identity(self):
        light = build_default_system_prompt(
            app_name="X", developer_name="D", attribution="A",
            flags=PromptFlags(intent="greeting"),
        )
        assert "identity" in light.lower() or "X" in light


# ---------------------------------------------------------------------------
# Fix #6: Context heuristic — _needs_conversation_context
# ---------------------------------------------------------------------------

class TestNeedsConversationContext:
    def setup_method(self):
        self.svc = _make_svc()
        self.history = [{"question": "What is TCP?", "answer": "TCP is..."}]

    def test_no_context_needed_for_standalone(self):
        assert not self.svc._needs_conversation_context(
            "What is a binary tree?", self.history
        )

    def test_no_context_without_history(self):
        assert not self.svc._needs_conversation_context("explain more", None)
        assert not self.svc._needs_conversation_context("explain more", [])

    def test_context_needed_for_explain_more(self):
        assert self.svc._needs_conversation_context("explain more", self.history)

    def test_context_needed_for_short_followup(self):
        assert self.svc._needs_conversation_context("and?", self.history)
        assert self.svc._needs_conversation_context("yes", self.history)

    def test_context_needed_for_demonstrative_pronoun(self):
        assert self.svc._needs_conversation_context(
            "What about this approach?", self.history
        )

    def test_no_false_positive_for_compare(self):
        """'Compare TCP vs UDP' should NOT need context — it's standalone."""
        assert not self.svc._needs_conversation_context(
            "Compare TCP vs UDP", self.history
        )

    def test_no_false_positive_for_why(self):
        """'Why is normalization important?' should NOT need context."""
        assert not self.svc._needs_conversation_context(
            "Why is normalization important in databases?", self.history
        )

    def test_no_false_positive_for_change(self):
        """'How do you change a tire' should NOT need context."""
        assert not self.svc._needs_conversation_context(
            "How do you change a database schema?", self.history
        )

    def test_continuation_word(self):
        assert self.svc._needs_conversation_context(
            "Also explain the time complexity", self.history
        )


# ---------------------------------------------------------------------------
# Fix #8: _is_ambiguous — word count based
# ---------------------------------------------------------------------------

class TestIsAmbiguous:
    def setup_method(self):
        self.svc = _make_svc()

    def test_tcp_vs_udp_not_ambiguous(self):
        """'TCP vs UDP' (3 words) should NOT be flagged as ambiguous."""
        assert not self.svc._is_ambiguous("TCP vs UDP")

    def test_single_word_ambiguous(self):
        assert self.svc._is_ambiguous("polymorphism")

    def test_empty_ambiguous(self):
        assert self.svc._is_ambiguous("")

    def test_full_question_not_ambiguous(self):
        assert not self.svc._is_ambiguous("What is a binary search tree?")


# ---------------------------------------------------------------------------
# Fix #7: _has_sufficient_context — inverted semantics
# ---------------------------------------------------------------------------

class TestHasSufficientContext:
    def setup_method(self):
        self.svc = _make_svc()

    def test_empty_history_means_standalone(self):
        """With no history, ANY question is standalone (sufficient context)."""
        assert self.svc._has_sufficient_context("what about that?", [])

    def test_standalone_question(self):
        history = [{"question": "What is TCP?", "answer": "TCP is..."}]
        result = self.svc._has_sufficient_context(
            "What is a binary search tree?", history
        )
        assert result is True

    def test_follow_up_detected(self):
        history = [{"question": "What is TCP?", "answer": "TCP is..."}]
        result = self.svc._has_sufficient_context("it seems slow", history)
        assert result is False


# ---------------------------------------------------------------------------
# Fix #12: classify_is_technical — heuristic (no LLM)
# ---------------------------------------------------------------------------

class TestClassifyIsTechnical:
    @pytest.mark.asyncio
    async def test_code_block_detected(self):
        svc = _make_svc()
        is_tech, conf, cat = await svc.classify_is_technical(
            "Review my code", "```python\nprint('hello')\n```"
        )
        assert is_tech is True
        assert conf == 1.0
        assert cat == "code block detected"

    @pytest.mark.asyncio
    async def test_system_design_keyword(self):
        svc = _make_svc()
        is_tech, conf, cat = await svc.classify_is_technical(
            "design a scalable system", "The system uses a load balancer..."
        )
        assert is_tech is True
        assert conf >= 0.8

    @pytest.mark.asyncio
    async def test_greeting_not_technical(self):
        svc = _make_svc()
        is_tech, conf, cat = await svc.classify_is_technical(
            "hello how are you", "I'm doing well!"
        )
        assert is_tech is False
        assert conf < 0.5

    @pytest.mark.asyncio
    async def test_no_llm_call_needed(self):
        """Ensure classify_is_technical works with no API key (pure heuristic)."""
        svc = _make_svc()
        is_tech, conf, cat = await svc.classify_is_technical(
            "implement a binary search algorithm", "Here is the code..."
        )
        assert is_tech is True


# ---------------------------------------------------------------------------
# Fix #13: Topic-aware recommendations
# ---------------------------------------------------------------------------

class TestTopicAwareRecommendations:
    def setup_method(self):
        self.svc = _make_svc()

    def test_system_design_recommendations(self):
        recs = self.svc._auto_recommendations_for_question(
            "Design a URL shortener system", "Here is the architecture..."
        )
        assert len(recs) == 3
        assert any("scaling" in r.lower() or "trade-off" in r.lower() for r in recs)

    def test_coding_recommendations(self):
        recs = self.svc._auto_recommendations_for_question(
            "Implement a binary search algorithm", "Here is the code..."
        )
        assert len(recs) == 3
        assert any("edge case" in r.lower() or "complexity" in r.lower() for r in recs)

    def test_behavioral_recommendations(self):
        recs = self.svc._auto_recommendations_for_question(
            "Tell me about a time you showed leadership", "In my previous role..."
        )
        assert len(recs) == 3
        assert any("star" in r.lower() for r in recs)

    def test_default_recommendations(self):
        recs = self.svc._auto_recommendations_for_question(
            "What is polymorphism?", "Polymorphism is..."
        )
        assert len(recs) == 3

    def test_empty_returns_nothing(self):
        assert self.svc._auto_recommendations_for_question("", "answer") == []
        assert self.svc._auto_recommendations_for_question("q", "") == []


# ---------------------------------------------------------------------------
# Fix #11: user_tier threading
# ---------------------------------------------------------------------------

class TestUserTierThreading:
    def test_get_optimal_token_limit_accepts_user_tier(self):
        svc = _make_svc()
        # Should not raise; user_tier is an accepted parameter
        result_free = svc._get_optimal_token_limit("What is TCP?", 4096, user_tier="free")
        result_pro = svc._get_optimal_token_limit("What is TCP?", 4096, user_tier="pro")
        assert result_free < result_pro, "Pro tier should get higher budget than free"

    def test_generate_answer_signature_has_user_tier(self):
        """generate_answer should accept user_tier keyword argument."""
        import inspect
        svc = _make_svc()
        sig = inspect.signature(svc.generate_answer)
        assert "user_tier" in sig.parameters

    def test_stream_answer_signature_has_user_tier(self):
        """stream_answer should accept user_tier keyword argument."""
        import inspect
        svc = _make_svc()
        sig = inspect.signature(svc.stream_answer)
        assert "user_tier" in sig.parameters


# ---------------------------------------------------------------------------
# Fix #19: Identity word-boundary matching
# ---------------------------------------------------------------------------

class TestIdentityWordBoundary:
    def test_you_matches_as_word(self):
        """'you' should match as whole word, not substring of 'youtube'."""
        from app.services.llm.identity import identity_response_text
        settings = MagicMock()
        settings.app_name = "TestApp"
        settings.app_developer_name = "TestDev"
        settings.app_developer_attribution = ""

        # "who developed you" should trigger identity
        result = identity_response_text(settings, "who developed you")
        assert "TestApp" in result

    def test_youtube_does_not_trigger_ownership_branch(self):
        """'youtube' should NOT trigger the ownership/creator attribution branch."""
        from app.services.llm.identity import identity_response_text
        settings = MagicMock()
        settings.app_name = "TestApp"
        settings.app_developer_name = "TestDev"
        settings.app_developer_attribution = ""

        # "who owns youtube" should NOT produce the ownership-specific response
        # (the one saying "TestApp is an interview preparation assistant...")
        result = identity_response_text(settings, "who owns youtube")
        # The ownership branch says "For development/ownership details, refer to official documentation."
        # If \byou\b incorrectly matched "youtube", this would appear.
        assert "For development/ownership details" not in result


# ---------------------------------------------------------------------------
# Fix #10: PromptFlags intent field
# ---------------------------------------------------------------------------

class TestPromptFlagsIntent:
    def test_prompt_flags_has_intent(self):
        flags = PromptFlags(intent="greeting")
        assert flags.intent == "greeting"

    def test_default_intent_is_empty(self):
        flags = PromptFlags()
        assert flags.intent == ""

    def test_flags_from_plan_sets_intent(self):
        svc = _make_svc()
        plan = svc._build_response_plan("hello there")
        flags = svc._flags_from_plan(plan, "hello there")
        assert hasattr(flags, "intent")
        assert isinstance(flags.intent, str)
