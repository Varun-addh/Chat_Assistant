"""
Interviewer Agent - Production-grade implementation.
Manages 5-question interview sequence with TTS and micro-feedback.
"""

import logging
from typing import List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import asyncio

from app.schemas import (
    PracticeInterviewQuestion,
    QuestionDifficulty,
    MicroFeedback,
    SpeechMetrics,
    SpeechAnalyticsConfig
)

logger = logging.getLogger(__name__)

# Try to import Gemini for AI analysis (optional)
try:
    from app.services.chat.gemini_adapter import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Gemini not available - using rule-based analysis only")


class InterviewerAgent:
    """
    Agent 1: Interviewer Agent
    Manages question flow, TTS generation, and micro-feedback.
    """
    
    # Curated question bank for realistic practice
    QUESTION_BANK = {
        QuestionDifficulty.EASY: [
            {
                "text": "Tell me about yourself and your background.",
                "category": "behavioral",
                "time_limit": 90
            },
            {
                "text": "Why are you interested in this position?",
                "category": "behavioral",
                "time_limit": 90
            },
            {
                "text": "What are your greatest strengths?",
                "category": "behavioral",
                "time_limit": 90
            },
            {
                "text": "Describe a time when you worked effectively in a team.",
                "category": "behavioral",
                "time_limit": 90
            },
            {
                "text": "Where do you see yourself in five years?",
                "category": "behavioral",
                "time_limit": 90
            }
        ],
        QuestionDifficulty.MEDIUM: [
            {
                "text": "Describe a challenging project you worked on and how you overcame obstacles.",
                "category": "behavioral",
                "time_limit": 120
            },
            {
                "text": "Tell me about a time when you had to learn a new technology quickly. How did you approach it?",
                "category": "technical",
                "time_limit": 120
            },
            {
                "text": "How do you handle conflicting priorities and tight deadlines?",
                "category": "behavioral",
                "time_limit": 90
            },
            {
                "text": "Describe a situation where you had to give difficult feedback to a colleague.",
                "category": "behavioral",
                "time_limit": 120
            },
            {
                "text": "What's your approach to debugging a complex issue in production?",
                "category": "technical",
                "time_limit": 120
            }
        ],
        QuestionDifficulty.HARD: [
            {
                "text": "Describe a time when you failed at something important. What did you learn and how did you apply those lessons?",
                "category": "behavioral",
                "time_limit": 150
            },
            {
                "text": "Tell me about a time when you had to make a critical decision without all the information you needed. How did you approach it?",
                "category": "behavioral",
                "time_limit": 150
            },
            {
                "text": "Describe a situation where you had to influence stakeholders who disagreed with your technical approach.",
                "category": "technical",
                "time_limit": 150
            },
            {
                "text": "How would you design a system to handle a sudden 10x increase in traffic? Walk me through your thought process.",
                "category": "system_design",
                "time_limit": 180
            },
            {
                "text": "Tell me about a time when you had to balance technical debt against feature delivery. How did you make the decision?",
                "category": "technical",
                "time_limit": 150
            }
        ]
    }
    
    def __init__(self, analytics_config: SpeechAnalyticsConfig, gemini_api_key: str = None):
        """
        Initialize the interviewer agent.
        
        Args:
            analytics_config: Configuration for analytics
            gemini_api_key: Optional Gemini API key for AI analysis
        """
        self.analytics_config = analytics_config
        self.gemini_model = None
        
        # Initialize Gemini if available and API key provided
        if GEMINI_AVAILABLE and gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-3-flash-preview')
                logger.info("Interviewer Agent initialized with AI analysis")
            except Exception as e:
                logger.warning(f"Gemini initialization failed: {e}")
        else:
            logger.info("Interviewer Agent initialized (rule-based only)")
    
    def get_questions(
        self, 
        difficulty: QuestionDifficulty = QuestionDifficulty.MEDIUM,
        count: int = 5
    ) -> List[PracticeInterviewQuestion]:
        """
        Get a set of interview questions.
        
        Args:
            difficulty: Question difficulty level
            count: Number of questions (default 5)
            
        Returns:
            List of InterviewQuestion objects
        """
        import random
        questions_data = list(self.QUESTION_BANK.get(difficulty, self.QUESTION_BANK[QuestionDifficulty.MEDIUM]))
        
        # Shuffle to ensure variety on every new session
        shuffled_data = list(questions_data)
        random.shuffle(shuffled_data)
        
        # Take first 'count' questions
        selected = shuffled_data[:count]
        
        # Convert to PracticeInterviewQuestion objects
        questions = [
            PracticeInterviewQuestion(
                id=i + 1,
                text=q["text"],
                difficulty=difficulty,
                time_limit=q["time_limit"],
                category=q["category"]
            )
            for i, q in enumerate(selected)
        ]
        
        logger.info(f"Generated {len(questions)} randomized questions at {difficulty} difficulty")
        return questions
    
    async def analyze_answer_quality(
        self,
        question: str,
        transcript: str,
        metrics: SpeechMetrics
    ) -> tuple[str, str]:
        """
        Use Gemini AI to analyze speech quality and content relevance.
        Falls back to rules if AI unavailable.
        """
        if not self.gemini_model:
            # Fallback to rule-based
            if metrics.confidence_score >= 0.6 and metrics.filler_count <= 2:
                speech_quality = "Clear delivery"
            else:
                speech_quality = "Keep practicing"
            
            if metrics.duration < 20:
                content_relevance = "Too brief"
            elif metrics.duration > 180:
                content_relevance = "Too long"
            else:
                content_relevance = "Good structure"
            
            return speech_quality, content_relevance
        
        try:
            prompt = f"""Analyze this interview answer briefly.

Question: {question}
Answer: {transcript}

Metrics: {metrics.wpm} WPM, {metrics.filler_count} fillers, {metrics.confidence_score:.2f} confidence

Provide (max 8 words each):
SPEECH_QUALITY: <assessment>
CONTENT_RELEVANCE: <assessment>"""

            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt,
                generation_config={"temperature": 0.3, "max_output_tokens": 100}
            )
            
            text = response.text.strip()
            speech_quality = "Good delivery"
            content_relevance = "Relevant answer"
            
            for line in text.split('\n'):
                if 'SPEECH_QUALITY:' in line:
                    speech_quality = line.split('SPEECH_QUALITY:')[1].strip()
                elif 'CONTENT_RELEVANCE:' in line:
                    content_relevance = line.split('CONTENT_RELEVANCE:')[1].strip()
            
            return speech_quality, content_relevance
            
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            return "Good effort", "Keep practicing"
    
    async def generate_micro_feedback(
        self,
        metrics: SpeechMetrics,
        question_text: str = "",
        transcript: str = ""
    ) -> MicroFeedback:
        """
        Generate 1-2 micro-feedback tips based on metrics.
        
        Rules:
        - Maximum 2 tips
        - Each < 15 words
        - Delivery-focused (not content judgment)
        - Prioritized by severity
        
        Args:
            metrics: Speech metrics from the answer
            
        Returns:
            MicroFeedback object with tips
        """
        tips = []
        
        # Priority 1: Excessive filler words
        if metrics.filler_count > 7:
            tips.append(f"Try reducing filler words — you used {metrics.filler_count}.")
        elif metrics.filler_count > 5:
            tips.append(f"Watch filler words — you used {metrics.filler_count} this time.")
        
        # Priority 2: Speaking pace issues
        if metrics.wpm > 180 and len(tips) < 2:
            tips.append("You're speaking too fast — slow down a little.")
        elif metrics.wpm < 120 and len(tips) < 2:
            tips.append("Try speaking a bit faster for confidence.")
        
        # Priority 3: Long pauses
        if metrics.longest_silence > 4 and len(tips) < 2:
            tips.append(f"You paused {metrics.longest_silence:.1f} seconds — keep the flow.")
        elif metrics.longest_silence > 3 and len(tips) < 2 and not tips:
            tips.append("Try to minimize long pauses.")
        
        # Priority 4: Overtalking
        if metrics.overtalked and len(tips) < 2:
            tips.append("Try to keep your answers more concise.")
        
        # Priority 5: Low confidence
        if metrics.confidence_score < 4 and len(tips) < 2:
            tips.append("Speak with more steady tone for confidence.")
        
        # If no issues found, give encouragement
        if not tips:
            tips.append("Great delivery — keep it up!")
        
        # Ensure max 2 tips
        tips = tips[:2]
        
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
        
        # Generate overall note
        if metrics.filler_count == 0 and 120 <= metrics.wpm <= 160:
            overall_note = "Excellent delivery!"
        elif metrics.filler_count < 3 and metrics.longest_silence < 2:
            overall_note = "Strong answer delivery"
        elif metrics.filler_count > 7 or metrics.longest_silence > 4:
            overall_note = "Focus on smoother delivery"
        else:
            overall_note = "Good effort, keep practicing"
        
        # Get AI-powered analysis if available
        if question_text and transcript:
            speech_quality, content_relevance = await self.analyze_answer_quality(
                question_text, transcript, metrics
            )
        else:
            # Fallback to basic assessment
            if metrics.confidence_score >= 0.6:
                speech_quality = "Clear delivery"
            else:
                speech_quality = "Practice clarity"
            content_relevance = "Unable to assess without transcript"
        
        logger.info(f"Generated micro-feedback: {len(tips)} tips, AI quality={speech_quality}")
        return MicroFeedback(
            delivery_tips=tips,
            pace_feedback=pace_feedback,
            overall_note=overall_note,
            speech_quality=speech_quality,
            content_relevance=content_relevance
        )
    
    def get_progress_indicator(self, current: int, total: int = 5) -> str:
        """
        Get progress indicator string.
        
        Args:
            current: Current question number (1-based)
            total: Total questions
            
        Returns:
            Progress string like "3/5"
        """
        return f"{current}/{total}"
    
    def format_tts_text(
        self, 
        question: PracticeInterviewQuestion, 
        total_questions: int = 5,
        company_name: Optional[str] = None
    ) -> str:
        """
        🚀 WORLD-CLASS INTERVIEWER CONVERSATIONAL ENGINE 🚀
        
        Transforms raw question text into a natural, high-stakes interview dialogue.
        Simulates real human interviewer patterns: fillers, intros, state-transitions.
        """
        import random
        
        # Personas for variety
        # 1. Professional (Standard technical interviewer)
        # 2. Friendly (Warm, encouraging coach)
        # 3. Direct (Fast-paced, executive style)
        persona = random.choice(["professional", "friendly", "direct"])
        
        # Determine company context
        company_segment = f" at {company_name}" if company_name and company_name.lower() not in ["any", "generic", "startup", "enterprise"] else ""
        
        # --- 1. INTRO SEGMENT (Shortened for speed) ---
        if question.id == 1:
            if persona == "friendly":
                intros = [
                    f"Hi! Glad to be speaking with you today{company_segment}.",
                    f"Welcome. Let's start our conversation with this.",
                    f"Hi! To get us started,"
                ]
            elif persona == "direct":
                intros = [
                    f"Alright, let's jump right in.",
                    f"Okay, first thing's first{company_segment}.",
                    f"Thanks for joining. Let's start."
                ]
            else: # professional
                intros = [
                    f"Hello. Thanks for your time today{company_segment}.",
                    f"Good to meet you. Let's start with your background. ",
                    f"Alright. Let's begin the session."
                ]
        elif question.id == total_questions:
            intros = [
                "Alright, we're on our final question.",
                "To wrap things up,",
                "And finally, one last question.",
                "Last one before we finish."
            ]
        else:
            # mid-session transitions (snappy)
            intros = [
                "Got it. Next,",
                "Alright. Moving on,",
                "Interesting. Now,",
                "Okay, follow-up question.",
                "Understood. Now,"
            ]

        intro = random.choice(intros)
        
        # --- 2. THE QUESTION ---
        question_text = self._add_natural_pauses(question.text)
        
        # --- 3. THE "INTERVIEWER" FLAVOR (Reduced frequency for speed) ---
        curiosity_fillers = [
            "I'm curious,",
            "Can you tell me,",
            "Could you share,",
            "I'd like to know,"
        ]
        
        # Lower probability (30%) to keep it fast
        if random.random() > 0.7 and not question_text.lower().startswith(("can you", "could you", "tell me", "explain")):
            filler = random.choice(curiosity_fillers)
            question_text = f"{filler} {question_text}"

        # --- 4. CONTEXTUAL PROMPTS (POST-QUESTION - Simplified) ---
        post_prompt = ""
        category = question.category.lower()
        
        if category in ['behavioral', 'situational']:
            prompts = [
                "... Looking for a specific past example here.",
                "... Feel free to use the STAR method.",
                "... I'd like to hear about the situation and outcome."
            ]
            post_prompt = f" {random.choice(prompts)}"
            
        elif category == 'technical':
            if question.difficulty == QuestionDifficulty.HARD:
                prompts = [
                    "... Take a moment to think this through.",
                    "... Interested in your thought process here."
                ]
                post_prompt = f" {random.choice(prompts)}"
        
        # Combine everything
        formatted = f"{intro} {question_text}{post_prompt}"
        
        # Final cleanup
        formatted = " ".join(formatted.split())
        
        return formatted
    
    def _add_natural_pauses(self, text: str) -> str:
        """
        Intelligently add natural pauses to text for better TTS intonation.
        
        Uses punctuation and strategic pauses to make TTS sound more natural:
        - Commas create brief pauses
        - Periods create longer pauses
        - Ellipsis (...) creates thinking pauses
        - Question marks trigger rising intonation in most TTS engines
        
        Args:
            text: Original question text
            
        Returns:
            Text with enhanced punctuation for natural delivery
        """
        import re
        
        # Ensure question ends with proper punctuation
        text = text.strip()
        
        # Comprehensive list of question indicators (verbs, question words, etc.)
        question_patterns = [
            # Question words
            r'^\s*what\b', r'^\s*how\b', r'^\s*why\b', r'^\s*when\b', r'^\s*where\b', 
            r'^\s*who\b', r'^\s*which\b', r'^\s*whose\b', r'^\s*whom\b',
            
            # Auxiliary verb questions
            r'^\s*can you\b', r'^\s*could you\b', r'^\s*would you\b', r'^\s*will you\b',
            r'^\s*do you\b', r'^\s*did you\b', r'^\s*have you\b', r'^\s*has\b',
            r'^\s*are you\b', r'^\s*were you\b', r'^\s*is there\b', r'^\s*are there\b',
            r'^\s*should you\b', r'^\s*may i\b', r'^\s*might you\b',
            
            # Imperative that expects response (question-like)
            r'^\s*tell me\b', r'^\s*describe\b', r'^\s*explain\b', r'^\s*discuss\b',
            r'^\s*share\b', r'^\s*give me\b', r'^\s*provide\b', r'^\s*walk me through\b'
        ]
        
        # Check if it's a question using regex patterns
        is_question = any(re.search(pattern, text, re.IGNORECASE) for pattern in question_patterns)
        
        if is_question and not text.endswith('?'):
            # Remove period if exists and add question mark
            text = text.rstrip('.!') + '?'
        
        # Dynamic transition phrase detection - add comma after imperative verbs
        imperative_phrases = [
            # Command/request phrases
            r'^\s*(Tell me about)\b', r'^\s*(Describe)\b', r'^\s*(Explain)\b', 
            r'^\s*(Discuss)\b', r'^\s*(Share)\b', r'^\s*(Walk me through)\b',
            r'^\s*(Talk about)\b', r'^\s*(Give me)\b', r'^\s*(Provide)\b',
            
            # Transitional phrases
            r'^\s*(In your opinion)\b', r'^\s*(For example)\b', r'^\s*(First)\b',
            r'^\s*(Finally)\b', r'^\s*(Additionally)\b', r'^\s*(Moreover)\b',
            
            # Question starters that need pause
            r'^\s*(Can you tell)\b', r'^\s*(Could you explain)\b', r'^\s*(Would you describe)\b'
        ]
        
        for pattern in imperative_phrases:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and ',' not in text[:match.end() + 5]:
                phrase = match.group(1)
                text = re.sub(re.escape(phrase), phrase + ',', text, count=1, flags=re.IGNORECASE)
                break
        
        # Add comma after "and" in long sentences for breathing room
        if len(text) > 80:
            # Find first "and" after 40 characters for mid-sentence pause
            parts = text.split(' and ', 1)
            if len(parts) == 2 and len(parts[0]) > 40:
                text = parts[0] + ' and,' + parts[1]
        
        # Add pause before "or" for clarity in choices
        text = re.sub(r'\s+or\s+', ' or, ', text)
        
        # Add comma before "but" for contrasts
        text = re.sub(r'\s+but\s+', ', but ', text)
        
        return text
    
    def validate_answer_timing(
        self, 
        duration: float, 
        time_limit: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate answer timing.
        
        Args:
            duration: Actual answer duration
            time_limit: Question time limit
            
        Returns:
            Tuple of (is_valid, warning_message)
        """
        if duration < 10:
            return False, "Answer too short (< 10 seconds)"
        
        if duration > time_limit * 1.5:
            return False, f"Answer too long (> {time_limit * 1.5:.0f} seconds)"
        
        return True, None
    
    def get_next_question(
        self, 
        questions: List[PracticeInterviewQuestion],
        current_index: int
    ) -> Optional[PracticeInterviewQuestion]:
        """
        Get the next question in sequence.
        
        Args:
            questions: List of all questions
            current_index: Current 0-based index (last answered question)
            
        Returns:
            Next question or None if complete
        """
        # current_index is the last answered question (0-based)
        # so next question is at current_index + 1
        next_index = current_index + 1
        if next_index < len(questions):
            return questions[next_index]
        return None
    
    def is_interview_complete(
        self, 
        current_index: int, 
        total_questions: int
    ) -> bool:
        """
        Check if interview is complete.
        
        Args:
            current_index: Current 0-based index (last answered question)
            total_questions: Total number of questions
            
        Returns:
            True if all questions answered
        """
        # current_index is 0-based, so if we answered question at index N-1, we're done
        return current_index >= total_questions - 1
    
    def get_question_context(self, question: PracticeInterviewQuestion) -> str:
        """
        Get context/category information for a question.
        
        Args:
            question: Interview question
            
        Returns:
            Context string
        """
        contexts = {
            "behavioral": "This is a behavioral question focusing on past experiences.",
            "technical": "This is a technical question about your skills and approach.",
            "system_design": "This is a system design question about architecture."
        }
        return contexts.get(question.category, "")


class MicroFeedbackEngine:
    """Advanced micro-feedback generation with smart prioritization."""
    
    def __init__(self, config: SpeechAnalyticsConfig):
        """Initialize with analytics config."""
        self.config = config
        self.feedback_rules = self._initialize_rules()
    
    def _initialize_rules(self) -> List[dict]:
        """Initialize feedback rules with priorities."""
        return [
            {
                "condition": lambda m: m.filler_count > 10,
                "message": lambda m: f"Reduce filler words significantly — {m.filler_count} is too many.",
                "priority": 1
            },
            {
                "condition": lambda m: m.filler_count > 7,
                "message": lambda m: f"Try reducing filler words — you used {m.filler_count}.",
                "priority": 2
            },
            {
                "condition": lambda m: m.wpm > 200,
                "message": lambda m: "You're speaking very fast — take a breath.",
                "priority": 1
            },
            {
                "condition": lambda m: m.wpm > 180,
                "message": lambda m: "You're speaking too fast — slow down a little.",
                "priority": 2
            },
            {
                "condition": lambda m: m.wpm < 100,
                "message": lambda m: "Try speaking faster for energy and confidence.",
                "priority": 2
            },
            {
                "condition": lambda m: m.wpm < 120,
                "message": lambda m: "Try speaking a bit faster for confidence.",
                "priority": 3
            },
            {
                "condition": lambda m: m.longest_silence > 5,
                "message": lambda m: f"Avoid very long pauses — {m.longest_silence:.1f}s is too long.",
                "priority": 2
            },
            {
                "condition": lambda m: m.longest_silence > 4,
                "message": lambda m: f"You paused {m.longest_silence:.1f} seconds — keep the flow.",
                "priority": 3
            },
            {
                "condition": lambda m: m.overtalked,
                "message": lambda m: "Try to keep your answers more concise.",
                "priority": 3
            },
            {
                "condition": lambda m: m.confidence_score < 3.5,
                "message": lambda m: "Speak with steadier tone for confidence.",
                "priority": 2
            },
            {
                "condition": lambda m: m.pause_count > 5,
                "message": lambda m: "Too many pauses — practice smoother delivery.",
                "priority": 3
            }
        ]
    
    def generate(self, metrics: SpeechMetrics) -> List[str]:
        """
        Generate prioritized feedback tips.
        
        Args:
            metrics: Speech metrics
            
        Returns:
            List of 0-2 feedback tips
        """
        applicable_tips = []
        
        # Check each rule
        for rule in self.feedback_rules:
            if rule["condition"](metrics):
                applicable_tips.append({
                    "message": rule["message"](metrics),
                    "priority": rule["priority"]
                })
        
        # Sort by priority
        applicable_tips.sort(key=lambda x: x["priority"])
        
        # Take top 2
        tips = [tip["message"] for tip in applicable_tips[:2]]
        
        # If no issues, provide encouragement
        if not tips:
            tips = ["Great delivery — keep it up!"]
        
        return tips
