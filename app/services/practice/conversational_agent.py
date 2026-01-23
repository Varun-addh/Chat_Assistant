"""
🚀 Conversational AI Agent - Zero-click interview onboarding.
Infers user profile from natural language conversation.
"""

import logging
import re
import asyncio
from typing import Optional, Dict, Any, Tuple

from app.config import settings
from app.schemas import QuestionDifficulty, UserProfile

logger = logging.getLogger(__name__)


class ConversationalAgent:
    """
    AI agent that understands conversational input and infers user profile.
    Replaces form-based configuration with natural language interaction.
    """

    def __init__(self, gemini_api_key: str):
        """Initialize conversational agent."""
        logger.info("🚀 Conversational Agent initialized (Universal Provider Support)")

    async def infer_profile_from_conversation(
        self,
        user_input: str,
        context: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Tuple[UserProfile, str, QuestionDifficulty, int]:
        """
        Use AI to extract user profile from conversational input.

        Args:
            user_input: Natural language from user (voice or text)
            context: Additional context (resume, past sessions, etc.)

        Returns:
            Tuple of (UserProfile, ai_response_message, difficulty, question_count)
        """
        try:
            prompt = self._build_inference_prompt(user_input, context)

            from app.services.chat.llm_service import llm_service

            text = await llm_service.generate_text(
                prompt=prompt,
                api_key=api_key,
                temperature=0.3,
                max_tokens=500,
            )

            if not text:
                logger.warning("⚠️ LLM response empty in profile inference, using fallback")
                raise ValueError("Empty LLM response")

            profile, difficulty, count, message = self._parse_ai_response(text, user_input)
            return profile, message, difficulty, count

        except Exception as e:
            logger.error(f"Error inferring profile: {e}")
            return (
                self._get_default_profile(),
                "I'll set up a general interview for you. Let's start!",
                QuestionDifficulty.MEDIUM,
                settings.default_question_count,
            )

    def _build_inference_prompt(self, user_input: str, context: Optional[str]) -> str:
        """Build prompt for profile inference."""
        prompt = (
            """You are an AI interview coach. Extract interview requirements from user input.

User Input: "{user_input}"
""".format(user_input=user_input)
        )

        if context:
            prompt += f"\nAdditional Context: {context}\n"

        prompt += f"""
Extract and respond in this EXACT format:

DOMAIN: <Software Engineer|Data Scientist|Product Manager|Frontend|Backend|Full Stack|ML Engineer|etc>
EXPERIENCE_YEARS: <number>
SKILLS: <comma-separated list>
JOB_ROLE: <specific role they're targeting>
COMPANY_PREFERENCE: <SPECIFIC company name like Google|Meta|Amazon|Microsoft|Netflix|Apple|Stripe|Airbnb|Uber|startup|any>
INTERVIEW_FOCUS: <technical|behavioral|both>
DIFFICULTY: <easy|medium|hard>
QUESTION_COUNT: <number between {settings.min_question_count}-{settings.max_question_count} - MUST extract exact number from user input like "2 questions" → 2, "3 questions" → 3>
AI_MESSAGE: <Friendly 1-sentence confirmation to the user>

CRITICAL RULES FOR QUESTION_COUNT:
- Extract the EXACT number user specifies: "N questions" → QUESTION_COUNT: N
- Match user's explicit count precisely - DO NOT modify or assume
- If NO number mentioned → QUESTION_COUNT: {settings.default_question_count} (default)
- Valid range: {settings.min_question_count}-{settings.max_question_count}

IMPORTANT:
- If user mentions a SPECIFIC company (Google, Meta, Amazon, etc.), use EXACT company name in COMPANY_PREFERENCE
- Extract EXACT company names from phrases like "preparing for Google", "Meta interview", "Amazon SDE"
- Question count MUST match user's explicit request - do NOT change it!

Examples of question count extraction:
- "generate X interview questions" → QUESTION_COUNT: X (use exact number user specified)
- "need Y hard questions" → QUESTION_COUNT: Y (use exact number user specified)
- "preparing for Google" (no count) → QUESTION_COUNT: {settings.default_question_count}
- "Senior SWE interview" (no count) → QUESTION_COUNT: {settings.default_question_count}

Example extraction format:

User: "I'm preparing for Senior SWE at Google"
DOMAIN: Backend Engineer
EXPERIENCE_YEARS: 6
SKILLS: Python, System Design, Distributed Systems, Leadership
JOB_ROLE: Senior Software Engineer
COMPANY_PREFERENCE: Google
INTERVIEW_FOCUS: technical
DIFFICULTY: hard
QUESTION_COUNT: {settings.default_question_count}
AI_MESSAGE: Perfect! I'll create challenging Google-style technical questions for Senior SWE level.

Now extract from the user's input above:
"""
        return prompt

    def _parse_ai_response(
        self, ai_text: str, original_input: str
    ) -> Tuple[UserProfile, QuestionDifficulty, int, str]:
        """Parse the LLM's structured response."""
        try:
            domain = self._extract_field(ai_text, "DOMAIN")
            experience_years = int(
                self._extract_field(
                    ai_text, "EXPERIENCE_YEARS", str(settings.default_experience_years)
                )
            )
            skills_str = self._extract_field(ai_text, "SKILLS", "general programming")
            skills = [s.strip() for s in skills_str.split(",")]
            job_role = self._extract_field(ai_text, "JOB_ROLE", domain)
            company_pref = self._extract_field(ai_text, "COMPANY_PREFERENCE", "any")
            focus = self._extract_field(ai_text, "INTERVIEW_FOCUS", "both")

            if focus.lower() == "both":
                focus_list = ["technical", "behavioral"]
            elif focus.lower() == "technical":
                focus_list = ["technical"]
            elif focus.lower() == "behavioral":
                focus_list = ["behavioral"]
            else:
                focus_list = ["technical", "behavioral"]

            difficulty_str = self._extract_field(ai_text, "DIFFICULTY", "medium")
            count_str = self._extract_field(
                ai_text, "QUESTION_COUNT", str(settings.default_question_count)
            )
            count = int(count_str)
            count = max(settings.min_question_count, min(settings.max_question_count, count))
            ai_message = self._extract_field(
                ai_text,
                "AI_MESSAGE",
                f"Got it! Preparing your {domain} interview now...",
            )

            logger.info(
                f"📊 Extracted from AI: domain={domain}, exp={experience_years}, difficulty={difficulty_str}, count={count}"
            )

            difficulty_map = {
                "easy": QuestionDifficulty.EASY,
                "medium": QuestionDifficulty.MEDIUM,
                "hard": QuestionDifficulty.HARD,
            }
            difficulty = difficulty_map.get(
                difficulty_str.lower(), QuestionDifficulty.MEDIUM
            )

            profile = UserProfile(
                domain=domain,
                experience_years=experience_years,
                skills=skills,
                job_role=job_role,
                company_preference=company_pref,
                interview_focus=focus_list,
            )

            logger.info(
                f"✅ Inferred profile: {domain}, {experience_years}yrs, {difficulty_str}, {count} questions"
            )
            return profile, difficulty, count, ai_message

        except Exception as e:
            logger.warning(f"Error parsing AI response: {e}")
            return self._get_fallback_from_input(original_input)

    def _extract_field(self, text: str, field: str, default: str = "") -> str:
        pattern = rf"{field}:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else default

    def _get_default_profile(self) -> UserProfile:
        return UserProfile(
            domain=settings.default_domain,
            experience_years=settings.default_experience_years,
            skills=["programming", "problem solving"],
            job_role=settings.default_domain,
            company_preference="any",
            interview_focus=["technical", "behavioral"],
        )

    def _get_fallback_from_input(
        self, user_input: str
    ) -> Tuple[UserProfile, QuestionDifficulty, int, str]:
        input_lower = user_input.lower()

        if any(word in input_lower for word in ["data scien", "ml", "machine learning"]):
            domain = "Data Scientist"
            skills = ["Python", "pandas", "machine learning"]
        elif any(word in input_lower for word in ["frontend", "react", "ui", "ux"]):
            domain = "Frontend Engineer"
            skills = ["JavaScript", "React", "CSS"]
        elif any(word in input_lower for word in ["backend", "api", "server"]):
            domain = "Backend Engineer"
            skills = ["Python", "APIs", "databases"]
        else:
            domain = "Software Engineer"
            skills = ["programming", "problem solving"]

        if any(word in input_lower for word in ["senior", "staff", "principal", "lead"]):
            experience = 6
            difficulty = QuestionDifficulty.HARD
        elif any(word in input_lower for word in ["junior", "entry", "graduate", "new grad"]):
            experience = 1
            difficulty = QuestionDifficulty.EASY
        else:
            experience = 3
            difficulty = QuestionDifficulty.MEDIUM

        profile = UserProfile(
            domain=domain,
            experience_years=experience,
            skills=skills,
            job_role=domain,
            company_preference="any",
            interview_focus=["technical", "behavioral"],
        )

        message = f"Setting up a {difficulty.value} level {domain} interview for you!"
        return profile, difficulty, settings.default_question_count, message
