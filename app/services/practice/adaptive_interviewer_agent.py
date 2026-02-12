"""
Adaptive Interviewer Agent - World-class real-time interview experience.
Dynamically generates questions based on user profile, domain, experience, and skills.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import numpy as np

from app.schemas import (
    PracticeInterviewQuestion,
    QuestionDifficulty,
    MicroFeedback,
    SpeechMetrics,
    UserProfile,
    InterviewRound,
    RoundConfig
)

logger = logging.getLogger(__name__)


class AdaptiveInterviewerAgent:
    """
    Intelligent Interviewer Agent that adapts to user profile.
    Generates contextual, relevant questions based on:
    - Domain (Python, Data Science, Frontend, etc.)
    - Experience level (0-3yrs: Junior, 3-7yrs: Mid, 7+: Senior)
    - Skills (specific technologies)
    - Job role and company preference
    """
    
    def __init__(self, gemini_api_key: str, gemini_model: str):
        """
        Initialize adaptive interviewer agent.
        """
        self.default_model = gemini_model
        logger.info(f"Adaptive Interviewer Agent initialized (Universal Provider Support)")
    
    async def evaluate_answer_comprehensively(
        self,
        question: str,
        transcript: str,
        metrics: SpeechMetrics,
        expected_answer: Optional[str] = None,
        key_points: Optional[List[str]] = None,
        question_category: str = "technical",
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        🌟 WORLD-CLASS ANSWER EVALUATION 🌟
        
        Provides comprehensive, intelligent feedback on answer correctness.
        
        Features:
        - Correctness scoring (0-100%)
        - Technical accuracy assessment
        - Key concepts identification (covered vs missed)
        - Strengths and improvement areas
        - Actionable suggestions
        - Real-time feedback
        
        Args:
            question: The interview question
            transcript: User's transcribed answer
            metrics: Speech metrics
            expected_answer: Optional expected answer template
            key_points: Optional list of key concepts to look for
            question_category: Question type (technical/behavioral/system_design)
            
        Returns:
            Dict with evaluation results
        """
        try:
            # Aggressive sanitization to prevent safety filter false positives
            sanitized_transcript = self._sanitize_for_gemini(transcript)
            sanitized_question = self._sanitize_for_gemini(question)
            
            # Build intelligent evaluation prompt
            key_points_str = "\\n".join([f"- {kp}" for kp in key_points]) if key_points else "No specific key points defined"
            expected_str = f"\\n\\nIDEAL ANSWER TEMPLATE:\\n{expected_answer}" if expected_answer else ""
            
            prompt = f"""EVALUATE THE FOLLOWING INTERVIEW ANSWER.
    
QUESTION: {sanitized_question}
CATEGORY: {question_category}
KEY CONCEPTS EXPECTED:
{key_points_str}{expected_str}

CANDIDATE'S TRANSCRIPT:
"{sanitized_transcript}"

SPEECH METRICS (For context only):
- WPM: {metrics.wpm}, Filler Count: {metrics.filler_count}, Confidence: {metrics.confidence_score:.2f}

EVALUATION REQUIREMENTS:
1. Provide a 'reasoning' string first explaining your logic.
2. 'correctness_score': 0-100 based on technical accuracy and depth.
3. 'technical_accuracy': One word only: "Excellent", "Good", "Fair", or "Poor".
4. 'key_points_covered' & 'key_points_missed': Arrays of strings.
5. 'strengths', 'improvement_areas', 'actionable_suggestions': Arrays of strings (max 2 each).
6. 'detailed_feedback': 1-2 sentences of professional assessment.
7. 'model_answer': A concise 100-200 word ideal answer showing EXACTLY how a top candidate would answer this question. Include specific technical terms, examples, and structure.

RETURN ONLY VALID JSON. DO NOT INCLUDE MARKDOWN BLOCKS.

{{
  "reasoning": "Analyze the answer quality relative to key points here...",
  "correctness_score": 0,
  "technical_accuracy": "Excellent",
  "key_points_covered": [],
  "key_points_missed": [],
  "strengths": [],
  "improvement_areas": [],
  "actionable_suggestions": [],
  "detailed_feedback": "",
  "model_answer": ""
}}"""

            # Use universal LLM service for provider-agnostic generation
            from app.services.chat.llm_service import llm_service
            
            try:
                text = await llm_service.generate_text(
                    prompt=prompt,
                    api_key=api_key,
                    json_mode=True,
                    temperature=0.2,
                    max_tokens=800
                )
                
                if not text:
                    logger.warning("⚠️ LLM response empty in evaluation, using fallback")
                    return self._fallback_evaluation(transcript, key_points, metrics)
                
                # Parse JSON output
                import json
                import re
                
                try:
                    evaluation = json.loads(text)
                except json.JSONDecodeError as je:
                    logger.warning(f"Initial JSON parse failed, trying repair: {je}")
                    # Try to extract JSON between braces
                    match = re.search(r'\{.*\}', text, re.DOTALL)
                    if match:
                        text = match.group(0)
                        text = self._repair_evaluation_json(text)
                        try:
                            evaluation = json.loads(text)
                        except Exception:
                            text = self._aggressive_json_repair(text)
                            try:
                                evaluation = json.loads(text)
                            except Exception:
                                return self._fallback_evaluation(transcript, key_points, metrics)
                    else:
                        return self._fallback_evaluation(transcript, key_points, metrics)
            
            except Exception as e:
                logger.error(f"Error in LLM evaluation: {e}")
                return self._fallback_evaluation(transcript, key_points, metrics)
            
            # Validate and normalize
            correctness_score = max(0, min(100, int(evaluation.get("correctness_score", 50))))
            
            return {
                "correctness_score": correctness_score,
                "technical_accuracy": evaluation.get("technical_accuracy", "Fair"),
                "key_points_covered": evaluation.get("key_points_covered", []),
                "key_points_missed": evaluation.get("key_points_missed", []),
                "strengths": evaluation.get("strengths", [])[:2],
                "improvement_areas": evaluation.get("improvement_areas", [])[:2],
                "actionable_suggestions": evaluation.get("actionable_suggestions", [])[:2],
                "is_correct": correctness_score >= 70,
                "detailed_feedback": evaluation.get("detailed_feedback", ""),
                "model_answer": evaluation.get("model_answer", ""),
            }
            
        except Exception as e:
            logger.error(f"Answer evaluation failed: {e}")
            return self._fallback_evaluation(transcript, key_points, metrics)
    
    def _repair_evaluation_json(self, text: str) -> str:
        """Specialized JSON repair for evaluation responses."""
        import re
        
        # CRITICAL: Collapse multiline strings into single lines
        # Pattern: "some text\n    more text" -> "some text more text"
        text = re.sub(r'"([^"]*?)\s*\n+\s*([^"]*?)"', r'"\1 \2"', text, flags=re.DOTALL)
        
        # Fix unterminated strings at line boundaries
        lines = text.split('\n')
        fixed_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            # Count unescaped quotes
            quote_count = stripped.count('"') - stripped.count('\\"')
            
            # If odd quotes and ends with comma/brace, add closing quote
            if quote_count % 2 != 0:
                if stripped.endswith(','):
                    stripped = stripped[:-1] + '",'
                elif stripped.endswith('}'):
                    stripped = stripped[:-1] + '"}'
                elif stripped.endswith(']'):
                    stripped = stripped[:-1] + '"]'
                else:
                    stripped += '"'
            
            fixed_lines.append(stripped)
        
        text = ' '.join(fixed_lines)
        
        # Remove trailing commas before closing brackets/braces
        text = re.sub(r',\s*([\]}])', r'\1', text)
        
        # Compress whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _fallback_evaluation(self, transcript: str, key_points: Optional[List[str]], metrics: SpeechMetrics) -> Dict[str, Any]:
        """Fallback evaluation using rule-based logic."""
        # Simple keyword matching
        covered = []
        missed = []
        
        if key_points:
            transcript_lower = transcript.lower()
            for kp in key_points:
                if any(word.lower() in transcript_lower for word in kp.split()):
                    covered.append(kp)
                else:
                    missed.append(kp)
        
        # Score based on coverage
        if key_points:
            coverage_ratio = len(covered) / len(key_points) if key_points else 0.5
            base_score = int(coverage_ratio * 100)
        else:
            # No key points, use duration and metrics
            if metrics.duration >= 30 and metrics.filler_count <= 5:
                base_score = 70
            elif metrics.duration >= 20:
                base_score = 60
            else:
                base_score = 40
        
        return {
            "correctness_score": base_score,
            "technical_accuracy": "Good" if base_score >= 70 else "Fair",
            "key_points_covered": covered,
            "key_points_missed": missed,
            "strengths": ["Answer provided"] if base_score >= 50 else [],
            "improvement_areas": ["Add more technical details"] if base_score < 70 else [],
            "actionable_suggestions": ["Cover all key concepts"] if missed else ["Good coverage"],
            "is_correct": base_score >= 70,
            "detailed_feedback": f"Answer score: {base_score}%. " + ("Good job!" if base_score >= 70 else "Needs improvement."),
            "model_answer": "",
        }
    
    async def analyze_answer_quality(
        self,
        question: str,
        transcript: str,

        metrics: SpeechMetrics,
        api_key: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Use Gemini AI to analyze speech quality and content relevance.
        
        Args:
            question: The interview question asked
            transcript: User's transcribed answer
            metrics: Speech metrics from analytics
            
        Returns:
            Tuple of (speech_quality, content_relevance)
        """
        try:
            # Sanitize transcript to avoid safety filter triggers
            # Remove potential problematic words that might trigger safety filters
            sanitized_transcript = transcript.replace("kill", "stop").replace("attack", "address").replace("exploit", "use")
            
            prompt = f"""Analyze this interview answer and provide brief assessments.

Question: {question}

Answer: {sanitized_transcript}

Speech Metrics:
- Words per minute: {metrics.wpm}
- Filler words: {metrics.filler_count}
- Confidence score: {metrics.confidence_score:.2f}
- Duration: {metrics.duration:.1f}s

Provide two brief assessments (each max 8 words):

1. SPEECH_QUALITY: Evaluate clarity, coherence, and communication effectiveness
2. CONTENT_RELEVANCE: Evaluate if answer addresses question and shows understanding

Format:
SPEECH_QUALITY: <assessment>
CONTENT_RELEVANCE: <assessment>"""

            # Provider-agnostic: use the shared llm_service (Groq/Gemini/etc)
            from app.services.chat.llm_service import llm_service

            text = await llm_service.generate_text(
                prompt=prompt,
                api_key=api_key,
                json_mode=False,
                temperature=0.2,
                max_tokens=120,
            )

            if not text:
                raise ValueError("Empty LLM response")

            text = text.strip()
            
            # Parse response
            speech_quality = "Good delivery"
            content_relevance = "Relevant answer"
            
            for line in text.split('\n'):
                if 'SPEECH_QUALITY:' in line:
                    speech_quality = line.split('SPEECH_QUALITY:')[1].strip()
                elif 'CONTENT_RELEVANCE:' in line:
                    content_relevance = line.split('CONTENT_RELEVANCE:')[1].strip()
            
            return speech_quality, content_relevance
            
        except Exception as e:
            logger.warning(f"Error analyzing answer quality: {e}")
            # Fallback to rule-based analysis
            if metrics.confidence_score >= 0.6 and metrics.filler_count <= 2:
                speech_quality = "Clear and confident delivery"
            elif metrics.filler_count > 7:
                speech_quality = "Reduce fillers for clarity"
            else:
                speech_quality = "Good effort, keep practicing"
            
            if metrics.duration < 20:
                content_relevance = "Too brief - add details"
            elif metrics.duration > 180:
                content_relevance = "Too long - be concise"
            else:
                content_relevance = "Well-structured answer"
            
            return speech_quality, content_relevance
    
    async def generate_adaptive_questions(
        self,
        user_profile: Optional[UserProfile],
        difficulty: QuestionDifficulty,
        count: int = 5,
        round_type: Optional[InterviewRound] = None,
        api_key: Optional[str] = None,
        previously_asked: Optional[List[str]] = None,
    ) -> List[PracticeInterviewQuestion]:
        """
        Generate interview questions tailored to user profile and round type.
        
        Args:
            user_profile: User's domain, experience, skills
            difficulty: Base difficulty level
            count: Number of questions
            round_type: Specific interview round (NEW - for round-based practice)
            previously_asked: Question texts from prior sessions (for cross-session dedup)
            
        Returns:
            List of contextually relevant questions
        """
        if not user_profile:
            # Fallback to generic questions if no profile
            return await self._generate_generic_questions(difficulty, count, round_type)

        # Determine interview level based on experience
        interview_level = self._determine_interview_level(user_profile.experience_years, difficulty)

        # Build intelligent prompt (with round context if specified)
        prompt = self._build_adaptive_prompt(user_profile, interview_level, count, round_type, previously_asked=previously_asked)

        try:
            round_info = f" for {round_type.value} round" if round_type else ""
            logger.info(
                f"Generating {count} adaptive questions{round_info} for {user_profile.domain} with {user_profile.experience_years}yrs experience"
            )

            response = await self._call_llm(prompt, api_key)
            questions = self._parse_questions(response, user_profile, difficulty, round_type)

            # ── Post-generation semantic dedup (cross-session + intra-batch) ──
            dedup_stats: dict = {
                "requested": count,
                "generated": len(questions),
                "history_size": len(previously_asked) if previously_asked else 0,
                "rejected_vs_past": 0,
                "rejected_vs_batch": 0,
                "retry_count": 0,
                "fallback_count": 0,
                "final_count": 0,
            }

            # Always run dedup — even without history we get intra-batch dedup
            questions, metrics = self._semantic_dedup(questions, previously_asked)
            dedup_stats["rejected_vs_past"] = metrics["rejected_vs_past"]
            dedup_stats["rejected_vs_batch"] = metrics["rejected_vs_batch"]

            # If too many were rejected (>40%), retry once with higher temperature
            if len(questions) < count * 0.6:
                dedup_stats["retry_count"] = 1
                logger.info(
                    f"Only {len(questions)}/{count} survived dedup — regenerating remainder"
                )
                shortfall = count - len(questions)
                already = list(previously_asked or []) + [q.text for q in questions]
                retry_prompt = self._build_adaptive_prompt(
                    user_profile, interview_level, shortfall, round_type,
                    previously_asked=already,
                )
                try:
                    retry_resp = await self._call_llm(retry_prompt, api_key)
                    retry_qs = self._parse_questions(retry_resp, user_profile, difficulty, round_type)
                    retry_qs, retry_metrics = self._semantic_dedup(retry_qs, already)
                    dedup_stats["rejected_vs_past"] += retry_metrics["rejected_vs_past"]
                    dedup_stats["rejected_vs_batch"] += retry_metrics["rejected_vs_batch"]
                    questions.extend(retry_qs)
                except Exception as retry_err:
                    logger.warning(f"Retry generation failed: {retry_err}")
            # ─────────────────────────────────────────────────────────────

            if len(questions) < count:
                remaining = count - len(questions)
                dedup_stats["fallback_count"] = remaining
                logger.warning(f"Only generated {len(questions)}/{count} questions, using fallback")
                generic = await self._generate_generic_questions(difficulty, remaining, round_type)
                questions.extend(generic[:remaining])

            dedup_stats["final_count"] = len(questions[:count])
            total_rejected = dedup_stats["rejected_vs_past"] + dedup_stats["rejected_vs_batch"]
            rejection_pct = (total_rejected / dedup_stats["generated"] * 100) if dedup_stats["generated"] else 0
            logger.info(
                f"[DEDUP METRICS] requested={dedup_stats['requested']} "
                f"generated={dedup_stats['generated']} "
                f"rejected_past={dedup_stats['rejected_vs_past']} "
                f"rejected_batch={dedup_stats['rejected_vs_batch']} "
                f"rejection_pct={rejection_pct:.1f}% "
                f"retries={dedup_stats['retry_count']} "
                f"fallbacks={dedup_stats['fallback_count']} "
                f"final={dedup_stats['final_count']} "
                f"history={dedup_stats['history_size']}"
            )

            return questions[:count]

        except Exception as e:
            logger.error(f"Adaptive question generation failed: {e}", exc_info=True)
            return await self._generate_generic_questions(difficulty, count, round_type)

    async def generate_follow_up_question(
        self,
        *,
        user_profile: Optional[UserProfile],
        difficulty: QuestionDifficulty,
        round_type: Optional[InterviewRound],
        previous_question: PracticeInterviewQuestion,
        transcript: str,
        micro_feedback: Optional[MicroFeedback] = None,
        already_asked: Optional[List[str]] = None,
        target_question_id: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> Optional[PracticeInterviewQuestion]:
        """Generate ONE drilling follow-up question based on the candidate's last answer.

        This is used to make Practice Mode feel like a real interviewer:
        - Ask for clarification
        - Drill on missed concepts
        - Increase depth where the answer was shallow

        Returns a PracticeInterviewQuestion or None on failure.
        """

        try:
            prompt = self._build_follow_up_prompt(
                user_profile=user_profile,
                difficulty=difficulty,
                round_type=round_type,
                previous_question=previous_question,
                transcript=transcript,
                micro_feedback=micro_feedback,
                already_asked=already_asked or [],
            )

            response = await self._call_llm(prompt, api_key)
            q = self._parse_single_question(
                response=response,
                difficulty=difficulty,
                round_type=round_type,
                target_question_id=target_question_id,
            )
            return q
        except Exception as e:
            logger.warning(f"Follow-up question generation failed: {e}")
            return None

    def _build_follow_up_prompt(
        self,
        *,
        user_profile: Optional[UserProfile],
        difficulty: QuestionDifficulty,
        round_type: Optional[InterviewRound],
        previous_question: PracticeInterviewQuestion,
        transcript: str,
        micro_feedback: Optional[MicroFeedback],
        already_asked: List[str],
    ) -> str:
        """Prompt for a single follow-up question (strict JSON object only)."""

        profile_block = ""
        if user_profile is not None:
            skills_str = ", ".join((user_profile.skills or [])[:8])
            focus_areas = ", ".join(user_profile.interview_focus or []) if user_profile.interview_focus else "(not specified)"
            role_context = f"{user_profile.job_role}" if user_profile.job_role else "(not specified)"
            company_context = f"{user_profile.company_preference}" if user_profile.company_preference else "(not specified)"
            profile_block = (
                "CANDIDATE PROFILE:\n"
                f"- Domain: {user_profile.domain}\n"
                f"- Experience: {user_profile.experience_years} years\n"
                f"- Role: {role_context}\n"
                f"- Company preference: {company_context}\n"
                f"- Key skills: {skills_str or '(none)'}\n"
                f"- Focus areas: {focus_areas}\n\n"
            )
            # Inject resume context for claim-based follow-up probing
            if getattr(user_profile, "resume_context", None):
                profile_block += self._build_resume_prompt_block(user_profile.resume_context) + "\n\n"

        round_block = ""
        if round_type is not None:
            round_block = f"ROUND: {round_type.value}\n\n"

        prev_key_points = getattr(previous_question, "key_points", None) or []
        prev_expected = getattr(previous_question, "expected_answer_template", None) or ""

        missed = []
        improvement = []
        strengths = []
        correctness = None
        if micro_feedback is not None:
            missed = micro_feedback.key_points_missed or []
            improvement = micro_feedback.improvement_areas or []
            strengths = micro_feedback.strengths or []
            correctness = micro_feedback.correctness_score

        asked = [t.strip() for t in (already_asked or []) if (t or "").strip()]
        asked_block = "\n".join([f"- {t}" for t in asked[-12:]]) if asked else "(none)"

        prev_category = getattr(previous_question, "category", "technical")
        prev_type = getattr(previous_question, "question_type", None)
        prev_type_str = str(prev_type.value) if prev_type is not None else "voice"

        prompt = f"""You are a world-class interviewer running a REALISTIC follow-up drill.

Goal: Ask ONE next question that directly follows from the candidate's last answer.

Constraints:
- The follow-up MUST be specific to what the candidate said (use their words/claims).
- If key concepts were missed, drill those first.
- If the answer was shallow, ask for depth (tradeoffs, edge cases, failure modes, complexity, metrics).
- Do NOT repeat earlier questions.
- Keep the question as a SINGLE sentence when possible.
- Output STRICT JSON ONLY (one JSON object). No markdown, no commentary.

{profile_block}{round_block}
PREVIOUS QUESTION:
{previous_question.text}

PREVIOUS QUESTION CONTEXT:
- category: {prev_category}
- question_type: {prev_type_str}
- expected key points: {prev_key_points if prev_key_points else '(none)'}
- expected answer template: {prev_expected if prev_expected else '(none)'}

CANDIDATE ANSWER (TRANSCRIPT):
"{(transcript or '').strip()}"

EVALUATION SIGNALS:
- correctness_score: {correctness if correctness is not None else 'N/A'}
- strengths: {strengths if strengths else '(none)'}
- key_points_missed: {missed if missed else '(none)'}
- improvement_areas: {improvement if improvement else '(none)'}

ALREADY ASKED QUESTIONS (avoid duplicates):
{asked_block}

Return EXACTLY one JSON object in this schema:
{{
  "text": "...",
  "category": "technical" | "behavioral" | "system_design",
  "question_type": "voice" | "coding" | "system_design",
  "time_limit": 90,
  "difficulty": "easy" | "medium" | "hard",
  "key_points": ["...", "..."],
  "expected_answer_template": "...",
  "programming_language": null
}}

Guidance:
- Prefer question_type=voice unless the follow-up truly requires writing code.
- Keep time_limit realistic: voice 60-180; coding 600-900; system_design 150-180.
"""

        return prompt

    def _parse_single_question(
        self,
        *,
        response: str,
        difficulty: QuestionDifficulty,
        round_type: Optional[InterviewRound],
        target_question_id: Optional[int],
    ) -> Optional[PracticeInterviewQuestion]:
        """Parse a single question JSON object into PracticeInterviewQuestion."""
        import json
        import re

        text = (response or "").strip()
        if not text:
            return None

        # Strip markdown fences if any
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        text = text.strip()

        # Extract first JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0).strip()

        try:
            data = json.loads(text)
        except Exception:
            # Best-effort repair: remove trailing commas
            text = re.sub(r",\s*([\]}])", r"\1", text)
            try:
                data = json.loads(text)
            except Exception:
                return None

        if not isinstance(data, dict):
            return None

        q_text = (data.get("text") or "").strip()
        if not q_text:
            return None

        category = (data.get("category") or "technical").strip()
        time_limit = int(data.get("time_limit") or 90)

        from app.schemas import QuestionType
        qtype_str = str(data.get("question_type") or "voice").lower()
        if qtype_str == "coding":
            qtype = QuestionType.CODING
        elif qtype_str == "system_design":
            qtype = QuestionType.SYSTEM_DESIGN
        else:
            qtype = QuestionType.VOICE

        diff_str = str(data.get("difficulty") or "").lower()
        if diff_str == "easy":
            q_diff = QuestionDifficulty.EASY
        elif diff_str == "hard":
            q_diff = QuestionDifficulty.HARD
        elif diff_str == "medium":
            q_diff = QuestionDifficulty.MEDIUM
        else:
            q_diff = difficulty

        key_points = data.get("key_points")
        if isinstance(key_points, list):
            key_points = [str(x).strip() for x in key_points if str(x).strip()][:5]
        else:
            key_points = None

        expected = (data.get("expected_answer_template") or "").strip() or None
        programming_language = data.get("programming_language")
        if programming_language is not None:
            programming_language = str(programming_language).strip() or None

        return PracticeInterviewQuestion(
            id=int(target_question_id or 1),
            text=q_text,
            difficulty=q_diff,
            time_limit=max(30, int(time_limit)),
            category=category,
            question_type=qtype,
            programming_language=programming_language,
            key_points=key_points if key_points else None,
            expected_answer_template=expected,
            round_type=round_type,
        )
    
    def _determine_interview_level(self, experience_years: int, base_difficulty: QuestionDifficulty) -> str:
        """
        Determine interview complexity based on experience and difficulty.
        
        Returns: Level description for prompt
        """
        if experience_years <= 2:
            level = "Junior/Entry-level"
            focus = "fundamentals, basic problem-solving, learning ability"
        elif experience_years <= 5:
            level = "Mid-level"
            focus = "practical experience, design decisions, collaboration"
        elif experience_years <= 10:
            level = "Senior"
            focus = "architecture, leadership, complex problem-solving, mentorship"
        else:
            level = "Staff/Principal"
            focus = "strategic thinking, system design, cross-team impact, technical vision"
        
        # Adjust based on difficulty
        if base_difficulty == QuestionDifficulty.HARD:
            return f"{level} (challenging round - deep technical/behavioral depth)"
        elif base_difficulty == QuestionDifficulty.EASY:
            return f"{level} (screening round - core competencies)"
        else:
            return f"{level} (main round - comprehensive assessment)"

    @staticmethod
    def _format_previously_asked(previously_asked: Optional[List[str]]) -> str:
        """Format previously asked questions for the prompt."""
        if not previously_asked:
            return ""  # Caller decides whether to include the block at all
        # Limit to 30 to keep prompt size reasonable
        items = previously_asked[:30]
        lines = [f"  {i+1}. {q}" for i, q in enumerate(items)]
        footer = f"\n  ... and {len(previously_asked) - 30} more" if len(previously_asked) > 30 else ""
        return "\n".join(lines) + footer

    def _build_dedup_variety_block(self, previously_asked: Optional[List[str]], domain: str) -> str:
        """Build the dedup + variety hint block for the prompt.

        If there are no previously asked questions, emit a lightweight variety
        hint only.  When history exists, include the full dedup instruction.
        """
        ts = datetime.now().isoformat()
        if previously_asked:
            formatted = self._format_previously_asked(previously_asked)
            return (
                f"**PREVIOUSLY ASKED QUESTIONS (DO NOT REPEAT THESE):**\n"
                f"{formatted}\n\n"
                f"**VARIETY HINT (DYNAMISM):**\n"
                f"Current Session Context: {ts}\n"
                f"Ensure these questions are COMPLETELY DIFFERENT from the previously asked "
                f"questions listed above. Focus on different sub-topics, edge cases, or "
                f"specific architectural tradeoffs within {domain} that haven't been covered "
                f"yet. Avoid the most common \"textbook\" questions. If the candidate has been "
                f"asked about topic X before, ask about topic Y instead."
            )
        return (
            f"**VARIETY HINT (DYNAMISM):**\n"
            f"Current Session Context: {ts}\n"
            f"Generate fresh, unique questions. Focus on different sub-topics, edge cases, "
            f"or specific architectural tradeoffs within {domain}. Avoid the most common "
            f"\"textbook\" questions."
        )

    def _build_adaptive_prompt(
        self,
        profile: UserProfile,
        interview_level: str,
        count: int,
        round_type: Optional[InterviewRound] = None,
        previously_asked: Optional[List[str]] = None,
    ) -> str:
        """Build intelligent prompt for question generation with optional round context."""
        
        skills_str = ", ".join(profile.skills[:5])  # Top 5 skills
        focus_areas = ", ".join(profile.interview_focus) if profile.interview_focus else "general technical skills"
        
        # Build company-specific context
        if profile.company_preference and profile.company_preference.lower() not in ["any", "general"]:
            company_context = f" for {profile.company_preference}"
            company_style_note = self._get_company_interview_style(profile.company_preference)
        else:
            company_context = ""
            company_style_note = ""
        
        role_context = f" as a {profile.job_role}" if profile.job_role else ""
        
        # Build round-specific context
        round_context = self._get_round_specific_context(round_type) if round_type else ""
        round_focus = self._get_round_focus(round_type) if round_type else "general interview questions"

        # ── Resume context injection (claim-based probing) ───────────────
        resume_block = ""
        if getattr(profile, "resume_context", None):
            resume_block = self._build_resume_prompt_block(profile.resume_context)
        
        # Prepare fallback question mix (outside f-string to avoid backslash issues)
        fallback_mix = f"""- Behavioral/Situational: Real scenarios they've likely faced at this level
- Technical Deep-Dive: {profile.domain}-specific technical questions
- Problem-Solving: Relevant to their actual tech stack
- System Design (if senior+): Architecture decisions in their domain"""
        
        prompt = f"""You are a senior technical interviewer conducting a {interview_level} interview{role_context}{company_context}.

CANDIDATE PROFILE:
- Domain: {profile.domain}
- Experience: {profile.experience_years} years
- Key Skills: {skills_str}
- Focus Areas: {focus_areas}

{company_style_note}

{round_context}

{resume_block}

Generate {count} realistic interview questions that:
1. Match the candidate's experience level and domain
2. Test relevant skills from their profile
3. {round_focus}
4. {"Reflect " + profile.company_preference + "'s actual interview patterns" if profile.company_preference and profile.company_preference.lower() not in ["any", "general"] else "Reflect what top companies actually ask for this level"}
5. Progress from foundation → application → complex scenarios
6. **IMPORTANT**: Vary difficulty appropriately - mix of easy/medium/hard based on the interview level
{"7. **RESUME-BASED**: At least 30-40% of questions MUST probe specific claims, projects, or achievements from the candidate's resume. Ask them to explain HOW they achieved specific metrics, WHAT tradeoffs they made, and WHY they chose their approach." if resume_block else ""}

QUESTION MIX (distribute across these types):
{self._get_round_question_mix(round_type, profile.domain) if round_type else fallback_mix}

CRITICAL REQUIREMENTS:
- Make questions SPECIFIC to {profile.domain} and {skills_str}
- Use terminology and scenarios from their domain
- Questions should feel like a REAL {interview_level} interview
- Set REALISTIC time limits based on question complexity:
  * Simple questions (tell me about yourself, strengths): 60-90s
  * Standard behavioral/technical questions: 90-120s
  * Complex technical deep-dive questions: 120-150s
  * **CODING questions (write actual code): 600-900s (10-15 minutes)**
  * System design/architecture questions: 150-180s
  * If a question requires detailed explanation, give MORE time
  * Match time to what's ACTUALLY answerable in that duration

**QUESTION TYPE DETECTION (CRITICAL FOR UI):**
- Set "question_type" based on what the candidate needs to DO:
  * "voice" → Verbal answer (behavioral, verbal technical explanation)
  * "coding" → **Write actual code** (e.g., "Write Python code to...", "Implement a function...", "Write SQL query...")
  * "system_design" → Draw diagrams (architecture, system design)

**⚠️ CODING QUESTION DETECTION:**
If your question contains ANY of these phrases, set "question_type": "coding":
- "Write the code"
- "Write a function"
- "Implement"
- "Write Python/JavaScript/SQL"
- "Code snippet"
- "Write a program"
- "Create a function"
Example: "Write the Python code snippet to..." → question_type: "coding", time_limit: 600

**JSON FORMATTING RULES:**
1. Return ONLY a valid JSON array - no markdown, no code blocks
2. Each string MUST be properly closed with double quotes
3. Escape special characters in strings (use \\ for backslash, \\" for quotes)
4. No trailing commas after last item in arrays/objects
5. Keep strings on single lines - no line breaks within string values
6. Test your JSON is valid before returning

{self._build_dedup_variety_block(previously_asked, profile.domain)}

Format (strict JSON only - include key_points for answer evaluation):
[
  {{
    "text": "Your question here - keep it on one line",
    "category": "behavioral",
    "question_type": "voice",
    "time_limit": 90,
    "difficulty": "medium",
    "key_points": ["concept 1", "concept 2", "concept 3"],
    "expected_answer_template": "Brief outline of ideal answer"
  }},
  {{
    "text": "Write the Python code to calculate fibonacci sequence",
    "category": "technical",
    "question_type": "coding",
    "programming_language": "Python",
    "time_limit": 600,
    "difficulty": "medium",
    "key_points": ["recursion or iteration", "base cases", "efficiency"],
    "expected_answer_template": "Should use dynamic programming or memoization for efficiency"
  }},
  {{
    "text": "Design a scalable URL shortener system",
    "category": "system_design",
    "question_type": "system_design",
    "time_limit": 900,
    "difficulty": "hard",
    "key_points": ["database schema", "caching", "load balancing"],
    "expected_answer_template": "Should cover database, API design, and scalability"
  }}
]

**KEY_POINTS**: List 2-5 essential concepts/topics that MUST be covered for a correct answer.
**EXPECTED_ANSWER_TEMPLATE**: Brief 1-2 sentence outline of what a strong answer should include.
**QUESTION_TYPE**: Set to "coding" if question asks to write actual code, "system_design" for architecture, otherwise "voice".

IMPORTANT: Return ONLY the JSON array above. No explanations. No markdown. Just valid JSON.
Generate exactly {count} questions with proper formatting INCLUDING key_points, expected_answer_template, AND question_type."""

        return prompt

    # ── Resume context helpers ────────────────────────────────────────────

    @staticmethod
    def _build_resume_prompt_block(resume_context) -> str:
        """Build a compact prompt block from structured resume context for injection into prompts."""
        if resume_context is None:
            return ""

        # Accept both dict and Pydantic model
        if hasattr(resume_context, "model_dump"):
            rc = resume_context.model_dump()
        elif hasattr(resume_context, "dict"):
            rc = resume_context.dict()
        elif isinstance(resume_context, dict):
            rc = resume_context
        else:
            return ""

        lines = [
            "=== CANDIDATE RESUME CONTEXT (use for claim-based probing) ===",
        ]

        summary = rc.get("experience_summary", "")
        if summary and summary != "Not specified":
            lines.append(f"Summary: {summary}")

        roles = rc.get("role_titles", [])
        if roles:
            lines.append(f"Roles: {', '.join(roles[:5])}")

        skills = rc.get("skills", [])
        if skills:
            lines.append(f"Skills from resume: {', '.join(skills[:15])}")

        education = rc.get("education", "")
        if education and education != "Not specified":
            lines.append(f"Education: {education}")

        projects = rc.get("projects", [])
        if projects:
            lines.append("Key Projects:")
            for proj in projects[:5]:
                name = proj.get("name", "Unnamed") if isinstance(proj, dict) else getattr(proj, "name", "Unnamed")
                tech = proj.get("tech", []) if isinstance(proj, dict) else getattr(proj, "tech", [])
                claims = proj.get("claims", []) if isinstance(proj, dict) else getattr(proj, "claims", [])
                lines.append(f"  - {name} [{', '.join(tech[:5])}]")
                for claim in claims[:3]:
                    lines.append(f"    • Claim: {claim}")

        achievements = rc.get("achievements", [])
        if achievements:
            lines.append("Notable Achievements (probe these!):")
            for ach in achievements[:8]:
                lines.append(f"  • {ach}")

        lines.append("=== END RESUME CONTEXT ===")
        lines.append("")
        lines.append("RESUME-BASED PROBING GUIDANCE:")
        lines.append("- Ask the candidate to EXPLAIN how they achieved specific metrics from their resume")
        lines.append("- Probe the TRADEOFFS they made in their projects")
        lines.append("- Ask about FAILURE MODES and edge cases in their claimed systems")
        lines.append("- Verify DEPTH: 'You mentioned X — walk me through the architecture/implementation'")
        lines.append("- Challenge claims: 'You improved latency by Y% — what was the bottleneck and how did you measure it?'")

        return "\n".join(lines)
    
    async def _call_llm(self, prompt: str, api_key: Optional[str] = None) -> str:
        """Call LLM with JSON mode for structured output."""
        from app.services.chat.llm_service import llm_service
        
        try:
            text = await llm_service.generate_text(
                prompt=prompt,
                api_key=api_key,
                json_mode=True,
                temperature=0.8,
                max_tokens=4000
            )
            
            if not text:
                raise ValueError("LLM response empty")
                
            return text
            
        except Exception as e:
            logger.error(f"Error in LLM question generation: {e}")
            raise

    # ── Post-generation semantic dedup ────────────────────────────────────

    _embed_model = None  # class-level lazy singleton

    @classmethod
    def _get_embed_model(cls):
        """Lazy-load SentenceTransformer (reuses the model already on disk)."""
        if cls._embed_model is not None:
            return cls._embed_model
        try:
            from sentence_transformers import SentenceTransformer
            from pathlib import Path

            local_dir = str(Path("data/models/all-MiniLM-L6-v2").resolve())
            try:
                cls._embed_model = SentenceTransformer(local_dir)
            except Exception:
                cls._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            return cls._embed_model
        except Exception as e:
            logger.debug(f"SentenceTransformer unavailable, semantic dedup disabled: {e}")
            return None

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _semantic_dedup(
        self,
        questions: List[PracticeInterviewQuestion],
        previously_asked: Optional[List[str]],
        threshold: float = 0.82,
    ) -> tuple:
        """Remove questions semantically too similar to past OR to each other.

        Two-pass dedup:
          1. vs previously_asked  (cross-session)
          2. vs already-kept batch items  (intra-batch)

        Returns (kept_list, metrics_dict).
        metrics_dict keys: total, kept, rejected_vs_past, rejected_vs_batch,
                           rejection_details (list of dicts).
        """
        metrics: dict = {
            "total": len(questions),
            "kept": len(questions),
            "rejected_vs_past": 0,
            "rejected_vs_batch": 0,
            "rejection_details": [],
        }

        model = self._get_embed_model()
        if model is None:
            return questions, metrics  # graceful degradation

        try:
            new_texts = [q.text for q in questions]
            new_embeddings = model.encode(new_texts, convert_to_numpy=True, show_progress_bar=False)

            # --- Pass 1: reject vs previously asked ---
            past_embeddings = None
            if previously_asked:
                past_texts = previously_asked[:50]
                past_embeddings = model.encode(past_texts, convert_to_numpy=True, show_progress_bar=False)

            after_past: List[tuple] = []  # (question, embedding)
            for i, q in enumerate(questions):
                if past_embeddings is not None:
                    sims = [self._cosine_sim(new_embeddings[i], pe) for pe in past_embeddings]
                    max_sim = max(sims) if sims else 0.0
                    if max_sim >= threshold:
                        metrics["rejected_vs_past"] += 1
                        metrics["rejection_details"].append({
                            "text": q.text[:80], "reason": "vs_past", "sim": round(max_sim, 3),
                        })
                        logger.info(
                            f"Semantic dedup: rejected vs past (sim={max_sim:.3f}): {q.text[:80]}..."
                        )
                        continue
                after_past.append((q, new_embeddings[i]))

            # --- Pass 2: intra-batch pairwise dedup ---
            kept: List[PracticeInterviewQuestion] = []
            kept_embeddings: list = []
            for q, emb in after_past:
                if kept_embeddings:
                    batch_sims = [self._cosine_sim(emb, ke) for ke in kept_embeddings]
                    max_batch_sim = max(batch_sims)
                    if max_batch_sim >= threshold:
                        metrics["rejected_vs_batch"] += 1
                        metrics["rejection_details"].append({
                            "text": q.text[:80], "reason": "vs_batch", "sim": round(max_batch_sim, 3),
                        })
                        logger.info(
                            f"Semantic dedup: rejected intra-batch (sim={max_batch_sim:.3f}): {q.text[:80]}..."
                        )
                        continue
                kept.append(q)
                kept_embeddings.append(emb)

            metrics["kept"] = len(kept)
            total_rejected = metrics["rejected_vs_past"] + metrics["rejected_vs_batch"]
            if total_rejected:
                logger.info(
                    f"Semantic dedup: kept {len(kept)}/{len(questions)} "
                    f"(past={metrics['rejected_vs_past']}, batch={metrics['rejected_vs_batch']})"
                )
            return kept, metrics

        except Exception as e:
            logger.warning(f"Semantic dedup failed, returning all questions: {e}")
            return questions, metrics
    
    def _parse_questions(
        self,
        response: str,
        profile: UserProfile,
        difficulty: QuestionDifficulty,
        round_type: Optional[InterviewRound] = None
    ) -> List[PracticeInterviewQuestion]:
        """Parse Gemini response into question objects - now with JSON mode (guaranteed valid)."""
        import json
        import re
        
        # Clean response - remove markdown code blocks if any
        text = response.strip()
        
        # Extract JSON from markdown code blocks (fallback for older responses)
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        
        text = text.strip()
        
        # With JSON mode enabled, this should usually be valid JSON, but providers
        # can still occasionally return extra trailing content. Extract the first
        # balanced JSON array to avoid "Extra data" parse errors.
        extracted = self._extract_first_json_array(text)
        if extracted:
            text = extracted
        else:
            # Fallback: try to locate an array-looking block
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                text = match.group(0)
        
        try:
            # Direct parse - should work with JSON mode
            data = json.loads(text)
            logger.info(f"✅ JSON parsed successfully on first attempt (JSON mode working)")
        except json.JSONDecodeError as first_error:
            # If JSON mode is working, this should NEVER happen
            logger.warning(f"⚠️ JSON mode failed? Attempting repair: {first_error}")
            logger.debug(f"Problematic JSON near position {first_error.pos}: ...{text[max(0, first_error.pos-100):min(len(text), first_error.pos+100)]}...")
            
            # Try repair as fallback
            text = self._repair_json(text)
            extracted = self._extract_first_json_array(text)
            if extracted:
                text = extracted
            try:
                data = json.loads(text)
                logger.info(f"✅ JSON repaired successfully after basic cleanup")
            except json.JSONDecodeError as second_error:
                # Last resort: aggressive repair
                logger.warning(f"⚠️ Basic repair failed, trying aggressive repair")
                text = self._aggressive_json_repair(text)
                extracted = self._extract_first_json_array(text)
                if extracted:
                    text = extracted
                try:
                    data = json.loads(text)
                    logger.info(f"✅ JSON repaired with aggressive method")
                except json.JSONDecodeError as third_error:
                    logger.error(f"❌ All parsing attempts failed: {third_error}")
                    logger.error(f"Error at position {third_error.pos}: ...{text[max(0, third_error.pos-50):min(len(text), third_error.pos+50)]}...")
                    logger.debug(f"Full response (first 1000 chars): {response[:1000]}")
                    return []
        
        try:
            # JSON mode sometimes returns a wrapper object like {"questions": [...]}
            if isinstance(data, dict):
                for key in ("questions", "items", "data"):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break

            if not isinstance(data, list):
                logger.warning("Response is not a list")
                return []
            
            questions = []
            for idx, item in enumerate(data):
                # Some providers may return a list of strings. Accept it.
                if isinstance(item, str):
                    item = {"text": item}

                if not isinstance(item, dict):
                    continue
                
                # Extract fields with defaults
                question_text = (
                    item.get("text")
                    or item.get("question")
                    or item.get("question_text")
                    or item.get("prompt")
                    or ""
                )
                question_text = str(question_text).strip()
                if not question_text:
                    continue
                
                category = str(item.get("category", "technical") or "technical")
                # Be tolerant of strings like "90".
                try:
                    time_limit = int(item.get("time_limit", 90))
                except Exception:
                    time_limit = 90
                
                # NEW: Extract question type and coding-specific fields
                question_type_str = str(item.get("question_type", "voice") or "voice").lower()
                programming_language = item.get("programming_language", None)
                
                # Map string to enum
                from app.schemas import QuestionType
                if question_type_str == "coding":
                    question_type = QuestionType.CODING
                elif question_type_str == "system_design":
                    question_type = QuestionType.SYSTEM_DESIGN
                else:
                    question_type = QuestionType.VOICE
                
                # Parse difficulty from Gemini response (if provided), otherwise use user's selection
                question_difficulty_str = item.get("difficulty", "").lower()
                if question_difficulty_str == "easy":
                    question_difficulty = QuestionDifficulty.EASY
                elif question_difficulty_str == "hard":
                    question_difficulty = QuestionDifficulty.HARD
                elif question_difficulty_str == "medium":
                    question_difficulty = QuestionDifficulty.MEDIUM
                else:
                    # Fallback to user's requested difficulty
                    question_difficulty = difficulty
                
                # Extract key points and expected answer for evaluation
                key_points = item.get("key_points", [])
                if isinstance(key_points, str):
                    # Common failure mode: a comma-separated string.
                    key_points = [p.strip() for p in key_points.split(",") if p.strip()]
                if isinstance(key_points, list):
                    key_points = [str(x).strip() for x in key_points if str(x).strip()][:5]
                else:
                    key_points = []

                expected_answer = item.get("expected_answer_template")
                if not expected_answer:
                    expected_answer = item.get("expected")
                expected_answer = str(expected_answer or "")
                
                questions.append(PracticeInterviewQuestion(
                    id=idx + 1,
                    text=question_text,
                    difficulty=question_difficulty,
                    time_limit=max(30, time_limit),
                    category=category,
                    question_type=question_type,  # NEW - determines UI (voice/coding/system_design)
                    programming_language=programming_language,  # NEW - for coding questions
                    key_points=key_points if key_points else None,
                    expected_answer_template=expected_answer if expected_answer else None,
                    round_type=round_type
                ))
            
            return questions
            
        except Exception as e:
            logger.error(f"Failed to parse questions: {e}")
            # Log the problematic response for debugging
            logger.debug(f"Problematic response (first 500 chars): {response[:500]}")
            return []

    def _extract_first_json_array(self, text: str) -> Optional[str]:
        """Extract the first balanced JSON array from text.

        Handles common LLM failure modes:
        - leading/trailing prose
        - multiple arrays concatenated
        - valid array followed by extra tokens ("Extra data" JSONDecodeError)

        Returns the array substring (including brackets) or None.
        """

        if not text:
            return None

        start = text.find("[")
        if start < 0:
            return None

        in_string = False
        escape = False
        depth = 0

        for i in range(start, len(text)):
            ch = text[i]

            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch == "[":
                depth += 1
                continue
            if ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
                continue

        return None
    
    def _get_company_interview_style(self, company: str) -> str:
        """
        Get company-specific interview context dynamically using AI knowledge.
        Instead of hardcoding, let AI use its knowledge about the company.
        
        Handles both specific companies (Amazon, Google) and generic practice (any/general).
        """
        # Check if user wants generic practice (not company-specific)
        if company.lower() in ["any", "general", "any company", "none"]:
            return """TARGET: Generic Interview Practice (No specific company)

⚠️ CRITICAL: This is GENERAL interview practice, NOT company-specific!
- Frame questions generically: "If you were to join a company..." or "For a typical role in this field..."
- Use pronouns: "their mission", "their team", "their product" (NOT "our")
- Avoid specific company names or references
- Focus on universal interview principles and best practices
- Ask questions that apply across the industry

This is practice mode - help the candidate prepare for interviews at ANY company.
Focus on transferable skills, common patterns, and industry-standard expectations.
"""
        
        # Company-specific interview preparation
        return f"""TARGET COMPANY: {company}

⚠️ CRITICAL: The candidate is INTERVIEWING FOR {company}, NOT working there yet!
- Frame questions as: "If you were to join {company}..." or "For this role at {company}..."
- NEVER say: "As an employee at {company}..." or "In your role at {company}..."
- The candidate is an APPLICANT, not a current employee

Use your knowledge about {company}'s:
- Known interview patterns and processes
- Technical focus areas and tech stack
- Company culture and values
- Types of questions they typically ask
- Specific frameworks or methodologies they prefer
- Leadership principles or core values (if any)
- Real interview examples you're aware of

Tailor questions to match {company}'s actual interview style and expectations.
If this is a well-known company (Google, Meta, Amazon, etc.), reflect their specific patterns.
If this is a startup or lesser-known company, use general best practices for that company type.
"""
    
    def _get_round_specific_context(self, round_type: InterviewRound) -> str:
        """Get context specific to the interview round."""
        round_contexts = {
            InterviewRound.HR_SCREENING: """
🎯 ROUND TYPE: HR Screening / Initial Interview
FOCUS: Background verification, motivation assessment, culture fit evaluation
TYPICAL DURATION: 15-20 minutes
INTERVIEWER STYLE: HR professional, friendly but evaluating fit""",
            
            InterviewRound.TECHNICAL_ROUND_1: """
🎯 ROUND TYPE: Technical Round 1 - Fundamentals
FOCUS: Core technical concepts, basic problem-solving, domain fundamentals
TYPICAL DURATION: 30-45 minutes
INTERVIEWER STYLE: Technical engineer assessing foundational knowledge""",
            
            InterviewRound.TECHNICAL_ROUND_2: """
🎯 ROUND TYPE: Technical Round 2 - Deep Dive
FOCUS: Advanced concepts, architecture decisions, complex problem-solving
TYPICAL DURATION: 45-60 minutes
INTERVIEWER STYLE: Senior engineer evaluating depth and experience""",
            
            InterviewRound.SYSTEM_DESIGN: """
🎯 ROUND TYPE: System Design Round
FOCUS: Scalability, architecture, tradeoffs, distributed systems
TYPICAL DURATION: 45-60 minutes
INTERVIEWER STYLE: Architect/Principal engineer testing design thinking""",
            
            InterviewRound.BEHAVIORAL: """
🎯 ROUND TYPE: Behavioral/Leadership Round
FOCUS: STAR method scenarios, teamwork, conflict resolution, impact
TYPICAL DURATION: 30-40 minutes
INTERVIEWER STYLE: Hiring manager or team lead assessing soft skills""",
            
            InterviewRound.MANAGERIAL: """
🎯 ROUND TYPE: Managerial/Director Round
FOCUS: Strategic thinking, vision, cross-functional leadership
TYPICAL DURATION: 30-45 minutes
INTERVIEWER STYLE: Director/VP evaluating leadership potential""",
            
            InterviewRound.MACHINE_LEARNING: """
🎯 ROUND TYPE: Machine Learning Specialist Round
FOCUS: ML algorithms, model selection, feature engineering, deployment
TYPICAL DURATION: 45-60 minutes
INTERVIEWER STYLE: ML engineer/scientist testing ML expertise""",
            
            InterviewRound.DATA_ENGINEERING: """
🎯 ROUND TYPE: Data Engineering Round
FOCUS: Data pipelines, ETL, big data technologies, data modeling
TYPICAL DURATION: 45-60 minutes
INTERVIEWER STYLE: Data engineer evaluating data infrastructure knowledge""",
        }
        return round_contexts.get(round_type, "")
    
    def _get_round_focus(self, round_type: InterviewRound) -> str:
        """Get focus areas for specific round."""
        focuses = {
            InterviewRound.HR_SCREENING: "Focus on behavioral basics, motivation, and culture fit",
            InterviewRound.TECHNICAL_ROUND_1: "Focus on core technical fundamentals and problem-solving",
            InterviewRound.TECHNICAL_ROUND_2: "Focus on advanced technical depth and architecture",
            InterviewRound.SYSTEM_DESIGN: "Focus on scalability, distributed systems, and design tradeoffs",
            InterviewRound.BEHAVIORAL: "Focus on STAR method scenarios and soft skills",
            InterviewRound.MANAGERIAL: "Focus on strategic thinking and leadership",
            InterviewRound.MACHINE_LEARNING: "Focus on ML algorithms, models, and deployment",
            InterviewRound.DATA_ENGINEERING: "Focus on data pipelines and big data systems",
        }
        return focuses.get(round_type, "Focus on comprehensive technical and behavioral assessment")
    
    def _get_round_question_mix(self, round_type: InterviewRound, domain: str) -> str:
        """Get question mix specific to round type."""
        mixes = {
            InterviewRound.HR_SCREENING: """- Background & Experience: Tell me about yourself, your journey
- Motivation: Why this role? Why this company?
- Culture Fit: Work style, team collaboration
- Career Goals: Short-term and long-term aspirations""",
            
            InterviewRound.TECHNICAL_ROUND_1: f"""- Core Concepts: Fundamental {domain} principles
- Problem Solving: Basic coding/technical challenges
- Tools & Technologies: Experience with relevant tech stack
- Debugging: Approach to troubleshooting issues""",
            
            InterviewRound.TECHNICAL_ROUND_2: f"""- Advanced Concepts: Deep {domain} expertise
- Architecture: Design patterns and decisions
- Code Quality: Best practices, optimization
- Complex Scenarios: Real-world problem-solving""",
            
            InterviewRound.SYSTEM_DESIGN: """- Scalability: How to handle 10x, 100x traffic
- Distributed Systems: CAP theorem, consistency, availability
- Architecture: Microservices, monoliths, tradeoffs
- Real Systems: Design Twitter, Netflix, Uber""",
            
            InterviewRound.BEHAVIORAL: """- Teamwork: Collaboration and conflict resolution
- Leadership: Influence without authority
- Challenges: Overcoming difficult situations (STAR method)
- Impact: Measuring and demonstrating results""",
            
            InterviewRound.MANAGERIAL: """- Strategy: Long-term thinking and planning
- Vision: Technical roadmap and direction
- Cross-functional: Working with product, design, business
- Mentorship: Growing and leading teams""",
        }
        return mixes.get(round_type, f"- Comprehensive mix of technical and behavioral questions for {domain}")
    
    def _repair_json(self, text: str) -> str:
        """Repair common JSON formatting issues from LLM responses."""
        import re
        
        # Remove any line breaks within string values first
        # This prevents strings from being split across lines
        text = re.sub(r'"([^"]*?)\n([^"]*?)"', r'"\1 \2"', text, flags=re.DOTALL)
        
        # Fix unterminated strings at end of lines
        lines = text.split('\n')
        repaired_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('//') or stripped.startswith('#'):
                continue  # Skip empty lines and comments
                
            # Count quotes (excluding escaped quotes)
            quote_count = stripped.count('"') - stripped.count('\\"')
            
            # If odd number of quotes, we have an unterminated string
            if quote_count % 2 != 0:
                # Find where to add the closing quote
                if stripped.endswith(','):
                    stripped = stripped[:-1] + '",'
                elif stripped.endswith('}'):
                    stripped = stripped[:-1] + '"}'
                elif stripped.endswith(']'):
                    stripped = stripped[:-1] + '"]'
                else:
                    stripped = stripped + '"'
            
            repaired_lines.append(stripped)
        
        text = '\n'.join(repaired_lines)
        
        # Remove trailing commas before closing brackets/braces
        text = re.sub(r',\s*([\]}])', r'\1', text)
        
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure proper spacing around brackets/braces
        text = text.replace('}{', '}, {').replace('][', '], [')
        
        return text
    
    def _aggressive_json_repair(self, text: str) -> str:
        """More aggressive JSON repair as fallback."""
        import re
        
        # STEP 1: Remove all line breaks within string values (most common issue)
        # Match strings that span multiple lines and collapse them
        text = re.sub(r'"([^"]*?)\s*\n+\s*([^"]*?)"', r'"\1 \2"', text, flags=re.DOTALL)
        
        # STEP 2: Fix missing commas between array elements
        # Pattern: "}  {" or "}{"  should be "}, {"
        text = re.sub(r'\}\s*\{', '}, {', text)
        # Pattern: "]  [" or "]["  should be "], ["
        text = re.sub(r'\]\s*\[', '], [', text)
        
        # STEP 3: Fix missing commas after closing braces/brackets before quotes
        # Pattern: } "field" should be }, "field"
        text = re.sub(r'\}\s*"', '}, "', text)
        text = re.sub(r'\]\s*"', '], "', text)
        
        # STEP 4: Fix broken strings across multiple lines more aggressively
        lines = text.split('\n')
        fixed_lines = []
        in_string = False
        current_string = ""
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty or comment lines
            if not stripped or stripped.startswith('//') or stripped.startswith('#'):
                continue
            
            # Count unescaped quotes
            quote_count = stripped.count('"') - stripped.count('\\"')
            
            if in_string:
                # We're inside a broken string, append to current
                current_string += " " + stripped
                # Check if this line closes the string
                if quote_count % 2 != 0:  # Odd means it closes
                    fixed_lines.append(current_string)
                    in_string = False
                    current_string = ""
            else:
                # Not in a string
                if quote_count % 2 != 0:  # Odd number = unterminated
                    # Check if this starts a multiline string
                    if not (stripped.endswith(',') or stripped.endswith('}') or stripped.endswith(']')):
                        # Likely a multiline string
                        in_string = True
                        current_string = stripped
                    else:
                        # Just add closing quote
                        if stripped.endswith(','):
                            fixed_lines.append(stripped[:-1] + '",')  
                        elif stripped.endswith('}'):
                            fixed_lines.append(stripped[:-1] + '"}')
                        elif stripped.endswith(']'):
                            fixed_lines.append(stripped[:-1] + '"]')
                        else:
                            fixed_lines.append(stripped + '"')
                else:
                    fixed_lines.append(stripped)
        
        text = ' '.join(fixed_lines)
        
        # STEP 5: Fix trailing commas
        text = re.sub(r',\s*([\]}])', r'\1', text)
        
        # STEP 6: Fix spacing and ensure proper structure
        text = re.sub(r'\s+', ' ', text)
        
        # STEP 7: Add missing commas between consecutive key-value pairs
        # Pattern: "key": "value" "nextkey": should be "key": "value", "nextkey":
        text = re.sub(r'"\s+"([a-zA-Z_][a-zA-Z0-9_]*)":', r'", "\1":', text)
        
        # STEP 8: Fix array elements without commas
        # Pattern: "text" } { should be "text" }, {
        text = re.sub(r'"\s*\}\s*\{', '" }, {', text)
        
        return text
    
    async def _generate_generic_questions(
        self,
        difficulty: QuestionDifficulty,
        count: int,
        round_type: Optional[InterviewRound] = None
    ) -> List[PracticeInterviewQuestion]:
        """Fallback: Generate generic questions without profile."""
        
        generic_bank = {
            QuestionDifficulty.EASY: [
                ("Tell me about yourself and your technical background.", "behavioral", 60),  # Simple intro
                ("Why are you interested in this role?", "behavioral", 75),
                ("Describe your experience with the technologies listed in the job description.", "technical", 90),
                ("What are your greatest technical strengths?", "behavioral", 75),
                ("Where do you see yourself in the next few years?", "behavioral", 90)
            ],
            QuestionDifficulty.MEDIUM: [
                ("Describe a challenging technical problem you solved recently.", "technical", 120),
                ("Tell me about a time you had to learn a new technology quickly.", "behavioral", 105),
                ("How do you approach debugging complex issues in production?", "technical", 135),
                ("Describe a situation where you had to make a tradeoff between speed and quality.", "behavioral", 120),
                ("Walk me through your development workflow from requirement to deployment.", "technical", 150)
            ],
            QuestionDifficulty.HARD: [
                ("Describe a time you made a significant architectural decision. What were the tradeoffs?", "system_design", 180),
                ("Tell me about a failure that taught you an important lesson.", "behavioral", 150),
                ("How would you design a system to handle 10x current traffic?", "system_design", 180),
                ("Describe a situation where you influenced others on a technical direction.", "behavioral", 135),
                ("Walk me through how you'd troubleshoot a critical production outage.", "technical", 165)
            ]
        }
        
        templates = list(generic_bank.get(difficulty, generic_bank[QuestionDifficulty.MEDIUM]))
        import random
        random.shuffle(templates)
        
        questions = []
        for idx, (text, category, time_limit) in enumerate(templates[:count]):
            questions.append(PracticeInterviewQuestion(
                id=idx + 1,
                text=text,
                difficulty=difficulty,
                time_limit=time_limit,
                category=category,
                round_type=round_type
            ))
        
        return questions
    
    async def generate_micro_feedback(
        self,
        metrics: SpeechMetrics,
        question_text: str = "",
        transcript: str = "",
        question_key_points: Optional[List[str]] = None,
        question_expected_answer: Optional[str] = None,
        question_category: str = "technical",
        api_key: Optional[str] = None
    ) -> MicroFeedback:
        """
        🌟 WORLD-CLASS FEEDBACK GENERATION 🌟
        
        Generate comprehensive feedback with:
        - Speech delivery tips
        - Answer correctness evaluation (NEW!)
        - Technical accuracy assessment (NEW!)
        - Strengths and improvements (NEW!)
        - Actionable suggestions (NEW!)
        
        Args:
            metrics: Speech analytics metrics
            question_text: The question asked
            transcript: User's answer transcript
            question_key_points: Key concepts to check
            question_expected_answer: Expected answer template
            question_category: Question type
            
        Returns:
            Comprehensive MicroFeedback with correctness evaluation
        """
        # PART 1: Delivery Tips (existing logic)
        tips = []
        
        # VAD Filter feedback - Show removed silence
        if metrics.silence_removed and metrics.silence_removed > 5:
            tips.append(f"⏸️ {metrics.silence_removed}s of long silences removed - practice continuous speaking")
        elif metrics.silence_removed and metrics.silence_removed > 2:
            tips.append(f"Reduce pauses ({metrics.silence_removed}s silence detected)")
        
        # Filler words
        if metrics.filler_count > 5:
            tips.append(f"Try to reduce filler words (used {metrics.filler_count} times)")
        elif metrics.filler_count == 0:
            tips.append("Excellent - no filler words!")
        
        # Speaking pace
        if metrics.wpm < 120:
            tips.append("Speak slightly faster for better engagement")
        elif metrics.wpm > 180:
            tips.append("Slow down slightly for clarity")
        elif 140 <= metrics.wpm <= 160:
            tips.append("Great speaking pace!")
        
        # Pauses
        if metrics.longest_silence > 5:
            tips.append("Long pauses detected - prepare answers mentally first")
        
        # Confidence
        if metrics.confidence_score >= 0.75:
            tips.append("Strong, confident delivery!")
        elif metrics.confidence_score < 0.4:
            tips.append("Practice to improve voice stability")
        
        # Ensure we have at least 1 tip
        if not tips:
            tips.append("Good delivery overall, keep practicing!")
        
        # Generate pace feedback
        if metrics.wpm > 180:
            pace_feedback = "Too fast - slow down for clarity"
        elif metrics.wpm < 100:
            pace_feedback = "Too slow - increase pace"
        elif metrics.wpm < 120:
            pace_feedback = "Slightly slow - could be faster"
        elif metrics.wpm > 160:
            pace_feedback = "Slightly fast - slow down a bit"
        else:
            pace_feedback = "Great pace!"
        
        # PART 2: COMPREHENSIVE ANSWER EVALUATION (NEW!)
        if question_text and transcript:
            try:
                # Use world-class evaluation
                evaluation = await self.evaluate_answer_comprehensively(
                    question=question_text,
                    transcript=transcript,
                    metrics=metrics,
                    expected_answer=question_expected_answer,
                    key_points=question_key_points,
                    question_category=question_category,
                    api_key=api_key
                )
                
                # Build overall note based on correctness
                score = evaluation["correctness_score"]
                if score >= 90:
                    overall_note = f"🌟 Exceptional answer! ({score}%)"
                elif score >= 70:
                    overall_note = f"✅ Strong answer! ({score}%)"
                elif score >= 50:
                    overall_note = f"👍 Good effort ({score}%)"
                else:
                    overall_note = f"📚 Needs improvement ({score}%)"
                
                # Get speech quality from old method for backward compatibility
                speech_quality = "Clear delivery" if metrics.confidence_score >= 0.6 else "Practice clarity"
                
                return MicroFeedback(
                    # Delivery feedback
                    delivery_tips=tips[:2],
                    pace_feedback=pace_feedback,
                    overall_note=overall_note,
                    speech_quality=speech_quality,
                    # NEW: Correctness evaluation fields
                    correctness_score=evaluation["correctness_score"],
                    technical_accuracy=evaluation["technical_accuracy"],
                    key_points_covered=evaluation["key_points_covered"],
                    key_points_missed=evaluation["key_points_missed"],
                    strengths=evaluation["strengths"],
                    improvement_areas=evaluation["improvement_areas"],
                    actionable_suggestions=evaluation["actionable_suggestions"],
                    is_correct=evaluation["is_correct"],
                    model_answer=evaluation.get("model_answer") or None,
                    # Backward compatibility
                    content_relevance=evaluation["detailed_feedback"]
                )
                
            except Exception as e:
                logger.error(f"Comprehensive evaluation failed: {e}")
                # Fallback to basic evaluation
                pass
        
        # FALLBACK: Basic feedback (no comprehensive evaluation)
        if metrics.filler_count == 0 and 120 <= metrics.wpm <= 160:
            overall_note = "Excellent delivery!"
        elif metrics.filler_count < 3 and metrics.longest_silence < 2:
            overall_note = "Strong delivery"
        else:
            overall_note = "Good effort"
        
        # Basic content assessment
        if metrics.confidence_score >= 0.6 and metrics.filler_count <= 2:
            speech_quality = "Clear and confident"
        else:
            speech_quality = "Keep practicing"
        
        if metrics.duration < 20:
            content_relevance = "Too brief - add details"
        elif metrics.duration > 180:
            content_relevance = "Too long - be concise"
        else:
            content_relevance = "Well-structured answer"
        
        return MicroFeedback(
            delivery_tips=tips[:2],
            pace_feedback=pace_feedback,
            overall_note=overall_note,
            speech_quality=speech_quality,
            content_relevance=content_relevance
            # Note: correctness fields will be None in fallback mode
        )
    
    def _sanitize_for_gemini(self, text: str) -> str:
        """
        Sanitize text to prevent Gemini safety filter false positives.
        
        Common false positive triggers in technical/coding content:
        - Words like "kill", "attack", "exploit", "hack", "inject"
        - These are legitimate in programming context but trigger safety filters
        """
        # Create a sanitization mapping (technical → safe)
        replacements = {
            # Process/system terms
            "kill": "terminate",
            "killed": "terminated",
            "killing": "terminating",
            
            # Security terms
            "attack": "test",
            "exploit": "utilize",
            "hack": "modify",
            "hacked": "modified",
            "hacking": "modifying",
            
            # Database/SQL terms
            "inject": "insert",
            "injection": "insertion",
            
            # Network terms
            "bomb": "overload",
            "malicious": "unexpected",
            "threat": "thread",  # CRITICAL: Often transcribed as threat
            "threats": "threads", # CRITICAL: Often transcribed as threats
            
            # Code terms that might trigger
            "die": "exit",
            "abort": "cancel",
            "fatal": "critical",
            
            # Aggressive language
            "destroy": "remove",
            "annihilate": "clear",
        }
        
        sanitized = text
        for trigger, replacement in replacements.items():
            # Case-insensitive replacement
            sanitized = sanitized.replace(trigger, replacement)
            sanitized = sanitized.replace(trigger.capitalize(), replacement.capitalize())
            sanitized = sanitized.replace(trigger.upper(), replacement.upper())
        
        return sanitized
