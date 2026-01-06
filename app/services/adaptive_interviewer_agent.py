"""
Adaptive Interviewer Agent - World-class real-time interview experience.
Dynamically generates questions based on user profile, domain, experience, and skills.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio

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
  "detailed_feedback": ""
}}"""

            # Use universal LLM service for provider-agnostic generation
            from app.services.llm_service import llm_service
            
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
                        except:
                            text = self._aggressive_json_repair(text)
                            try:
                                evaluation = json.loads(text)
                            except:
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
                "detailed_feedback": evaluation.get("detailed_feedback", "")
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
            "detailed_feedback": f"Answer score: {base_score}%. " + ("Good job!" if base_score >= 70 else "Needs improvement.")
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

            if api_key:
                genai.configure(api_key=api_key)

            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config={"temperature": 0.3, "max_output_tokens": 100},
                safety_settings=self.safety_settings
            )
            
            # Check if response was blocked
            if not response.text:
                logger.warning("⚠️ Gemini response blocked in analyze_answer_quality, using fallback")
                logger.warning(f"Safety ratings: {response.candidates[0].safety_ratings if response.candidates else 'N/A'}")
                raise ValueError("Response blocked by safety filters")
            
            text = response.text.strip()
            
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
        api_key: Optional[str] = None
    ) -> List[PracticeInterviewQuestion]:
        """
        Generate interview questions tailored to user profile and round type.
        
        Args:
            user_profile: User's domain, experience, skills
            difficulty: Base difficulty level
            count: Number of questions
            round_type: Specific interview round (NEW - for round-based practice)
            
        Returns:
            List of contextually relevant questions
        """
        if not user_profile:
            # Fallback to generic questions if no profile
            return await self._generate_generic_questions(difficulty, count, round_type)
        
        # Determine interview level based on experience
        interview_level = self._determine_interview_level(user_profile.experience_years, difficulty)
        
        # Build intelligent prompt (with round context if specified)
        prompt = self._build_adaptive_prompt(user_profile, interview_level, count, round_type)
        
        try:
            round_info = f" for {round_type.value} round" if round_type else ""
            logger.info(f"Generating {count} adaptive questions{round_info} for {user_profile.domain} with {user_profile.experience_years}yrs experience")
            
            response = await self._call_llm(prompt, api_key)
            questions = self._parse_questions(response, user_profile, difficulty, round_type)
            
            if len(questions) < count:
                logger.warning(f"Only generated {len(questions)}/{count} questions, using fallback")
                # Fill remaining with generic questions
                remaining = count - len(questions)
                generic = await self._generate_generic_questions(difficulty, remaining, round_type)
                questions.extend(generic[:remaining])
            
            return questions[:count]
            
        except Exception as e:
            logger.error(f"Adaptive question generation failed: {e}", exc_info=True)
            return await self._generate_generic_questions(difficulty, count, round_type)
    
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
    
    def _build_adaptive_prompt(
        self,
        profile: UserProfile,
        interview_level: str,
        count: int,
        round_type: Optional[InterviewRound] = None
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

Generate {count} realistic interview questions that:
1. Match the candidate's experience level and domain
2. Test relevant skills from their profile
3. {round_focus}
4. {"Reflect " + profile.company_preference + "'s actual interview patterns" if profile.company_preference and profile.company_preference.lower() not in ["any", "general"] else "Reflect what top companies actually ask for this level"}
5. Progress from foundation → application → complex scenarios
6. **IMPORTANT**: Vary difficulty appropriately - mix of easy/medium/hard based on the interview level

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

**VARIETY HINT (DYNAMISM):** 
Current Session Context: {datetime.now().isoformat()}
Ensure these questions are unique and varied. Focus on different sub-topics, edge cases, or specific architectural tradeoffs within {profile.domain} that haven't been covered yet. Avoid the most common "textbook" questions.

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
    
    async def _call_llm(self, prompt: str, api_key: Optional[str] = None) -> str:
        """Call LLM with JSON mode for structured output."""
        from app.services.llm_service import llm_service
        
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
        
        # With JSON mode enabled, this should ALWAYS be valid JSON
        # But keep basic extraction logic as fallback
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
            try:
                data = json.loads(text)
                logger.info(f"✅ JSON repaired successfully after basic cleanup")
            except json.JSONDecodeError as second_error:
                # Last resort: aggressive repair
                logger.warning(f"⚠️ Basic repair failed, trying aggressive repair")
                text = self._aggressive_json_repair(text)
                try:
                    data = json.loads(text)
                    logger.info(f"✅ JSON repaired with aggressive method")
                except json.JSONDecodeError as third_error:
                    logger.error(f"❌ All parsing attempts failed: {third_error}")
                    logger.error(f"Error at position {third_error.pos}: ...{text[max(0, third_error.pos-50):min(len(text), third_error.pos+50)]}...")
                    logger.debug(f"Full response (first 1000 chars): {response[:1000]}")
                    return []
        
        try:
            if not isinstance(data, list):
                logger.warning("Response is not a list")
                return []
            
            questions = []
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                
                # Extract fields with defaults
                question_text = item.get("text", "").strip()
                if not question_text:
                    continue
                
                category = item.get("category", "technical")
                time_limit = int(item.get("time_limit", 90))
                
                # NEW: Extract question type and coding-specific fields
                question_type_str = item.get("question_type", "voice").lower()
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
                expected_answer = item.get("expected_answer_template", "")
                
                questions.append(PracticeInterviewQuestion(
                    id=idx + 1,
                    text=question_text,
                    difficulty=question_difficulty,
                    time_limit=time_limit,
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
        text = re.sub(r'"\s+"([a-zA-Z_])":', r'", "\1":', text)
        
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
