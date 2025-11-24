import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class InterviewType(str, Enum):
    """Types of interview questions"""
    CODING = "coding"
    BEHAVIORAL = "behavioral"
    SYSTEM_DESIGN = "system_design"
    TECHNICAL = "technical"


class DifficultyLevel(str, Enum):
    """Difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EvaluationCriteria(BaseModel):
    """Evaluation criteria with scores"""
    correctness: float = Field(ge=0, le=10, description="Technical correctness")
    completeness: float = Field(ge=0, le=10, description="Answer completeness")
    clarity: float = Field(ge=0, le=10, description="Communication clarity")
    confidence: float = Field(ge=0, le=10, description="Delivery confidence")
    technical_depth: float = Field(ge=0, le=10, description="Technical understanding")


class UserAnswer(BaseModel):
    """User's answer to interview question"""
    answer_text: str
    time_taken_seconds: Optional[int] = None
    input_method: str = "text"  # text or voice
    code_solution: Optional[str] = None
    language: Optional[str] = None


class InterviewQuestion(BaseModel):
    """Interview question with metadata"""
    question_id: str
    question_text: str
    interview_type: InterviewType
    difficulty: DifficultyLevel
    topic: str
    expected_points: List[str] = Field(default_factory=list)
    sample_answer: Optional[str] = None
    evaluation_rubric: Optional[Dict[str, Any]] = None
    hints: List[str] = Field(default_factory=list, description="Progressive hints for the question")
    time_limit_minutes: Optional[int] = None


class DetailedFeedbackItem(BaseModel):
    """Enhanced feedback item with context and examples"""
    point: str = Field(description="The feedback point")
    explanation: str = Field(description="Why this matters")
    example: Optional[str] = Field(default=None, description="Specific example from user's answer")
    impact_level: str = Field(default="medium", description="critical|high|medium|low")
    what_to_add: Optional[str] = Field(default=None, description="For weaknesses - what should be included")


class AnswerComparison(BaseModel):
    """Side-by-side comparison of user answer vs ideal"""
    aspect: str = Field(description="What aspect is being compared")
    user_said: str = Field(description="What the user actually said")
    should_say: str = Field(description="What a strong answer would include")
    gap_explanation: str = Field(description="Why the difference matters")


class EvaluationResult(BaseModel):
    """AI evaluation of user's answer with detailed learning feedback"""
    overall_score: float = Field(ge=0, le=10, description="Overall score out of 10")
    criteria_scores: EvaluationCriteria
    
    # Basic feedback (backward compatible)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    missing_points: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)
    
    # ENHANCED: Detailed feedback with context
    detailed_strengths: List[DetailedFeedbackItem] = Field(default_factory=list, description="Strengths with explanations")
    detailed_weaknesses: List[DetailedFeedbackItem] = Field(default_factory=list, description="Weaknesses with what to add")
    
    # ENHANCED: Answer comparison
    answer_comparisons: List[AnswerComparison] = Field(default_factory=list, description="Side-by-side user vs ideal")
    
    # Performance summary
    performance_summary: str
    detailed_feedback: str
    
    # Rating category
    rating_category: str  # "Excellent", "Good", "Fair", "Needs Improvement"
    
    # Follow-up questions
    follow_up_questions: List[str] = Field(default_factory=list)
    
    # Model answer showing the ideal response
    model_answer: str = Field(default="", description="Well-structured ideal answer demonstrating best practices")
    
    # ENHANCED: Learning resources
    recommended_resources: List[str] = Field(default_factory=list, description="Links/topics to study")
    key_takeaways: List[str] = Field(default_factory=list, description="3-5 main points to remember")


class InterviewSession(BaseModel):
    """Complete interview session"""
    session_id: str
    user_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Questions and answers
    questions: List[InterviewQuestion] = Field(default_factory=list)
    answers: List[UserAnswer] = Field(default_factory=list)
    evaluations: List[EvaluationResult] = Field(default_factory=list)
    
    # Session stats
    current_question_index: int = 0
    total_questions: int = 5
    average_score: Optional[float] = None
    
    # Session type
    interview_type: InterviewType
    difficulty: DifficultyLevel
    
    # Enhanced features
    hints_used: Dict[str, int] = Field(default_factory=dict, description="Track hints used per question")
    question_start_times: Dict[str, datetime] = Field(default_factory=dict, description="Track when each question started")
    time_per_question: Dict[str, int] = Field(default_factory=dict, description="Time spent on each question in seconds")


class MockInterviewService:
    """
    Service to manage mock interview sessions
    """
    
    def __init__(self, llm_service, interview_intelligence_service):
        self.llm_service = llm_service
        self.interview_service = interview_intelligence_service
        self.active_sessions: Dict[str, InterviewSession] = {}
        
        # Session persistence
        self.sessions_file = Path("data/sessions/mock_interview_sessions.json")
        self.sessions_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing sessions from disk
        self._load_sessions()
        
        logger.info(f"Mock Interview Service initialized with {len(self.active_sessions)} active sessions")
    
    def _load_sessions(self):
        """Load active sessions from disk"""
        try:
            if self.sessions_file.exists():
                try:
                    with open(self.sessions_file, 'r') as f:
                        data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Corrupted sessions file, backing up and starting fresh: {e}")
                    # Backup the corrupted file
                    backup_file = self.sessions_file.with_suffix('.corrupted.json')
                    self.sessions_file.rename(backup_file)
                    logger.info(f"Backed up corrupted file to {backup_file}")
                    return
                
                # Convert dict back to InterviewSession objects
                for session_id, session_data in data.items():
                    try:
                        # Parse datetime strings
                        session_data['started_at'] = datetime.fromisoformat(session_data['started_at'])
                        if session_data.get('completed_at'):
                            session_data['completed_at'] = datetime.fromisoformat(session_data['completed_at'])
                        
                        # Parse question_start_times datetime strings
                        if 'question_start_times' in session_data and session_data['question_start_times']:
                            converted_times = {}
                            for k, v in session_data['question_start_times'].items():
                                if isinstance(v, str):
                                    try:
                                        converted_times[k] = datetime.fromisoformat(v)
                                    except:
                                        logger.warning(f"Failed to parse datetime: {v}")
                                elif isinstance(v, datetime):
                                    converted_times[k] = v
                            session_data['question_start_times'] = converted_times
                        
                        # Reconstruct session
                        session = InterviewSession(**session_data)
                        self.active_sessions[session_id] = session
                    except Exception as e:
                        logger.warning(f"Failed to load session {session_id}: {e}")
                        continue
                
                logger.info(f"Loaded {len(self.active_sessions)} sessions from disk")
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}", exc_info=True)
    
    def _save_sessions(self):
        """Save active sessions to disk"""
        try:
            # Convert sessions to JSON-serializable format
            data = {}
            for session_id, session in self.active_sessions.items():
                try:
                    session_dict = session.model_dump()
                    # Convert datetime to ISO format strings
                    session_dict['started_at'] = session.started_at.isoformat()
                    if session.completed_at:
                        session_dict['completed_at'] = session.completed_at.isoformat()
                    
                    # Convert question_start_times datetime values to ISO strings
                    if 'question_start_times' in session_dict and session_dict['question_start_times']:
                        converted_times = {}
                        for k, v in session_dict['question_start_times'].items():
                            if isinstance(v, datetime):
                                converted_times[k] = v.isoformat()
                            elif isinstance(v, str):
                                converted_times[k] = v
                            else:
                                logger.warning(f"Unexpected type for question_start_time: {type(v)}")
                        session_dict['question_start_times'] = converted_times
                    
                    data[session_id] = session_dict
                except Exception as e:
                    logger.error(f"Failed to serialize session {session_id}: {e}")
                    continue
            
            # Write to a temporary file first, then rename (atomic operation)
            temp_file = self.sessions_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Atomic rename
            temp_file.replace(self.sessions_file)
            
            logger.debug(f"Saved {len(data)} sessions to disk")
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}", exc_info=True)
    
    async def start_session(
        self,
        user_id: str,
        interview_type: InterviewType,
        difficulty: DifficultyLevel,
        num_questions: int = 5,
        topic: Optional[str] = None
    ) -> InterviewSession:
        """
        Start a new mock interview session
        
        Args:
            user_id: User identifier
            interview_type: Type of interview (coding, behavioral, etc.)
            difficulty: Difficulty level
            num_questions: Number of questions in session
            topic: Optional specific topic
        
        Returns:
            InterviewSession object
        """
        session_id = str(uuid.uuid4())
        
        # Generate questions for the session
        questions = await self._generate_session_questions(
            interview_type=interview_type,
            difficulty=difficulty,
            num_questions=num_questions,
            topic=topic
        )
        
        session = InterviewSession(
            session_id=session_id,
            user_id=user_id,
            started_at=datetime.utcnow(),
            questions=questions,
            total_questions=len(questions),
            interview_type=interview_type,
            difficulty=difficulty
        )
        
        self.active_sessions[session_id] = session
        
        # Persist to disk
        self._save_sessions()
        
        logger.info(f"Started mock interview session {session_id} for user {user_id}")
        
        return session
    
    async def get_current_question(self, session_id: str) -> Optional[InterviewQuestion]:
        """Get the current question for a session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        if session.current_question_index >= len(session.questions):
            return None
        
        current_question = session.questions[session.current_question_index]
        
        # Track when this question was started
        question_id = current_question.question_id
        if question_id not in session.question_start_times:
            session.question_start_times[question_id] = datetime.utcnow()
            self._save_sessions()
        
        return current_question
    
    async def get_hint(
        self,
        session_id: str,
        hint_level: int = 1
    ) -> Optional[str]:
        """
        Get a progressive hint for the current question
        
        Args:
            session_id: Session identifier
            hint_level: Level of hint (1=gentle nudge, 2=more specific, 3=detailed)
        
        Returns:
            Hint text or None
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        current_question = await self.get_current_question(session_id)
        if not current_question:
            return None
        
        question_id = current_question.question_id
        
        # Track hint usage
        if question_id not in session.hints_used:
            session.hints_used[question_id] = 0
        
        session.hints_used[question_id] += 1
        self._save_sessions()
        
        # If question has predefined hints, use them
        if current_question.hints and hint_level <= len(current_question.hints):
            return current_question.hints[hint_level - 1]
        
        # Otherwise, generate AI hint based on level
        hint_type = 'gentle nudge' if hint_level == 1 else 'specific guidance' if hint_level == 2 else 'detailed hint'
        expected_str = ', '.join(current_question.expected_points) if current_question.expected_points else 'N/A'
        
        hint_prompt = f"""Generate a hint for this interview question. Level {hint_level}/3 (1=gentle, 3=detailed).
        
Question: {current_question.question_text}
Type: {current_question.interview_type}
Difficulty: {current_question.difficulty}
Expected points: {expected_str}

Provide a {hint_type} without giving away the full answer.
Keep it under 100 words."""
        
        hint = await self.llm_service.generate_answer(hint_prompt)
        
        logger.info(f"Generated hint level {hint_level} for session {session_id}, question {question_id}")
        
        return hint
    
    async def submit_answer(
        self,
        session_id: str,
        answer: UserAnswer
    ) -> EvaluationResult:
        """
        Submit answer and get AI evaluation
        
        Args:
            session_id: Session identifier
            answer: User's answer
        
        Returns:
            EvaluationResult with scores and feedback
        """
        session = self.active_sessions.get(session_id)
        if not session:
            available_sessions = list(self.active_sessions.keys())
            logger.error(f"Session {session_id} not found. Available sessions: {available_sessions}")
            raise ValueError(f"Session {session_id} not found")
        
        # Get current question
        current_question = session.questions[session.current_question_index]
        question_id = current_question.question_id
        
        # Calculate time spent on this question
        if question_id in session.question_start_times:
            time_spent = (datetime.utcnow() - session.question_start_times[question_id]).total_seconds()
            session.time_per_question[question_id] = int(time_spent)
        
        # Evaluate the answer
        evaluation = await self._evaluate_answer(
            question=current_question,
            answer=answer
        )
        
        # Store answer and evaluation
        session.answers.append(answer)
        session.evaluations.append(evaluation)
        
        # Move to next question
        session.current_question_index += 1
        
        # Update session stats
        if session.current_question_index >= session.total_questions:
            await self._complete_session(session)
        
        # Persist to disk
        self._save_sessions()
        
        logger.info(f"Evaluated answer for session {session_id}, score: {evaluation.overall_score}")
        
        return evaluation
    
    async def _evaluate_answer(
        self,
        question: InterviewQuestion,
        answer: UserAnswer
    ) -> EvaluationResult:
        """
        Use LLM to evaluate user's answer
        """
        logger.info(f"Evaluating answer with LLM service (enabled={self.llm_service.enabled}, provider={self.llm_service._settings.llm_provider})")
        
        if not self.llm_service.enabled:
            # Fallback: basic evaluation
            logger.warning("LLM service not enabled, using basic evaluation fallback")
            return self._basic_evaluation(question, answer)
        
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(question, answer)
        
        try:
            # Get LLM evaluation with high token limit for detailed feedback
            # Override the default token limit to ensure complete response
            import asyncio
            
            # Temporarily increase max tokens for evaluation
            original_max_tokens = self.llm_service._settings.groq_max_tokens
            self.llm_service._settings.groq_max_tokens = 4000  # Increased to 4000 for complete detailed feedback with enhanced fields
            
            try:
                response = await self.llm_service.generate_answer(prompt)
            finally:
                # Restore original setting
                self.llm_service._settings.groq_max_tokens = original_max_tokens
            
            # Parse response
            evaluation = self._parse_evaluation_response(response, question, answer)
            
            return evaluation
        
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}", exc_info=True)
            return self._basic_evaluation(question, answer)
    
    def _build_evaluation_prompt(
        self,
        question: InterviewQuestion,
        answer: UserAnswer
    ) -> str:
        """Build prompt for LLM evaluation"""
        
        prompt = f"""Expert interviewer: Provide COMPREHENSIVE, LEARNING-FOCUSED evaluation.

Q: {question.question_text}
Type: {question.interview_type} | Difficulty: {question.difficulty} | Topic: {question.topic}

Candidate Answer: {answer.answer_text}
{f"Code: {answer.code_solution}" if answer.code_solution else ""}

Expected Points: {", ".join(question.expected_points) if question.expected_points else "N/A"}

EVALUATION GOALS:
1. Help the candidate LEARN, not just judge
2. Show SPECIFIC examples from their answer
3. Explain WHY things matter (impact)
4. Provide actionable improvement steps
5. Include a MODEL ANSWER to learn from
6. Suggest resources to study

JSON STRUCTURE - RETURN EXACTLY THIS:
{{
  "correctness": 8.5,
  "completeness": 7.0,
  "clarity": 9.0,
  "confidence": 8.0,
  "technical_depth": 7.5,
  
  "strengths": ["Mentioned API Gateway", "Clear explanation"],
  "weaknesses": ["No capacity math", "Missing observability"],
  "missing_points": ["Rate limiting", "Monitoring setup"],
  "improvement_suggestions": ["Calculate exact capacity", "Add metrics"],
  
  "detailed_strengths": [
    {{
      "point": "Identified core microservices components",
      "explanation": "Shows understanding that systems need clear service boundaries",
      "example": "You mentioned API Gateway and Service Mesh which are industry standard",
      "impact_level": "high"
    }}
  ],
  
  "detailed_weaknesses": [
    {{
      "point": "No capacity calculations provided",
      "explanation": "Interviewers expect you to show you can estimate scale requirements",
      "example": "You said scale based on load but gave no numbers",
      "impact_level": "critical",
      "what_to_add": "Calculate: 10K req/sec means 10 instances at 1K req/sec each. With 3x buffer = 30 instances"
    }},
    {{
      "point": "Missing observability strategy",
      "explanation": "Production systems require monitoring and alerting",
      "impact_level": "high",
      "what_to_add": "Mention Prometheus for metrics, Grafana for dashboards, and PagerDuty for alerts"
    }}
  ],
  
  "answer_comparisons": [
    {{
      "aspect": "Capacity Planning",
      "user_said": "The system should scale based on load",
      "should_say": "For 1M users with 10K peak requests/sec, need 10 instances at 1K req/sec each. With 3x safety buffer = 30 instances",
      "gap_explanation": "Specific numbers show you can actually architect production systems, not just theory"
    }}
  ],
  
  "performance_summary": "Solid understanding of microservices architecture with clear communication. Main gap is lack of quantitative analysis - need to add capacity math and observability details for senior-level answers.",
  
  "detailed_feedback": "Your answer demonstrates good grasp of microservices fundamentals. You correctly identified API Gateway and Service Mesh as key components, showing industry knowledge. However, to reach senior-level, you need to add: (1) Capacity calculations with actual numbers, (2) Observability strategy with specific tools, (3) Rate limiting approach. Practice doing back-of-envelope calculations and always mention monitoring in system design answers.",
  
  "rating_category": "Good",
  
  "follow_up_questions": ["How would you calculate exact capacity needs?", "What monitoring tools would you use?"],
  
  "model_answer": "For a notification system serving 1 million users, I would design as follows:\n\nCAPACITY PLANNING:\nAssuming 10% daily active users = 100K users. Peak load at 8PM = 50% concentrated in 1 hour = 50K notifications. That is 14 notifications/second average, 50/sec peak. Each service instance handles 100 req/sec, so need 1 instance normally, 3 instances for peak with 3x buffer = 6 instances.\n\nARCHITECTURE:\n- Message Queue: Kafka with 3 partitions for parallel processing\n- Worker Pool: 6 auto-scaling instances consuming from Kafka\n- Rate Limiting: Token bucket algorithm, 100 notifications/min per user\n- Database: PostgreSQL for user preferences, Redis for rate limit counters\n\nOBSERVABILITY:\n- Metrics: Prometheus tracking latency p99, throughput, error rate\n- Dashboards: Grafana showing queue depth and worker health\n- Alerts: PagerDuty if error rate > 1% or latency > 500ms\n\nThis ensures 99.9% uptime with proper capacity and monitoring.",
  
  "recommended_resources": ["Back-of-envelope calculations guide", "System Design Interview book", "Microservices observability patterns"],
  
  "key_takeaways": [
    "Always include specific capacity numbers in system design",
    "Observability is not optional - mention metrics, logs, alerts",
    "Rate limiting prevents system overload",
    "Use concrete examples not generic statements"
  ]
}}

CRITICAL RULES:
1. Return ONLY JSON - no markdown, no code fences, no extra text
2. ALL arrays use bracket syntax: ["item1", "item2"]
3. detailed_strengths and detailed_weaknesses MUST have nested objects
4. impact_level MUST be: critical, high, medium, or low
5. Give SPECIFIC examples from user's answer in "example" field
6. model_answer must be 200-400 words showing EXACTLY how to answer perfectly
7. No apostrophes - use "can not" not "can't"
8. CRITICAL: All strings must be single-line - use \\n for line breaks, NOT actual newlines
9. Do NOT put actual line breaks inside quoted strings - use \\n instead
10. Keep all JSON on properly formatted lines with proper escaping
11. NEVER use backslash-quote at the end of a string value (NO: "text\", YES: "text")
12. If you need quotes inside a string, escape them: \\" not just "
13. All string values must be properly closed with unescaped quote: "value" not "value\\"
14. NO BLANK LINES between JSON properties - keep it compact
"""
        
        return prompt
    
    def _parse_evaluation_response(
        self,
        response: str,
        question: InterviewQuestion,
        answer: UserAnswer
    ) -> EvaluationResult:
        """Parse LLM response into EvaluationResult"""
        
        # Clean response
        response = response.strip()
        
        # Log raw response for debugging
        logger.debug(f"Raw LLM evaluation response: {response[:500]}...")
        
        # Remove markdown code fences
        if "```json" in response:
            response = response.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in response:
            response = response.split("```", 1)[1].split("```", 1)[0]
        
        response = response.strip()
        
        # FIX: Remove invalid control characters (newlines, tabs in strings)
        # This is the main fix for "Invalid control character" errors
        import re
        import json as json_module
        
        # AGGRESSIVE FIX: Replace ALL control characters with escaped versions
        # This handles cases where the LLM puts literal newlines/tabs in string values
        def escape_control_chars(text: str) -> str:
            """Escape all control characters in the text"""
            # Replace control characters with their escaped equivalents
            replacements = {
                '\n': '\\n',
                '\r': '\\r',
                '\t': '\\t',
                '\b': '\\b',
                '\f': '\\f',
                '\v': '\\v',
                '\0': '\\0'
            }
            for char, escaped in replacements.items():
                text = text.replace(char, escaped)
            return text
        
        # Apply to the entire response
        response = escape_control_chars(response)
        
        # Now restore structural newlines (between top-level keys)
        # These are needed for JSON formatting but shouldn't be inside string values
        response = response.replace('\\n  ', '\n  ')  # Restore indentation
        response = response.replace('\\n}', '\n}')    # Restore closing braces
        response = response.replace('\\n{', '\n{')    # Restore opening braces  
        response = response.replace('{\\n', '{\n')    # After opening brace
        response = response.replace(',\\n', ',\n')    # After commas at top level
        response = response.replace('[\\n', '[\n')    # After opening bracket
        response = response.replace('\\n]', '\n]')    # Before closing bracket
        response = response.replace(': \\n', ': \n')  # After colons
        
        # CRITICAL FIX: The LLM is generating things like:
        # "point": "No answer provided\",
        # This is broken JSON - need to fix dangling backslash-quotes
        
        # Remove any backslash before quote-comma or quote-newline patterns
        response = re.sub(r'\\"\s*,', '",', response)  # Fix: \" , -> " ,
        response = re.sub(r'\\"\s*\n', '"\n', response)  # Fix: \" \n -> " \n
        response = re.sub(r'\\"\s*}', '"}', response)  # Fix: \" } -> " }
        response = re.sub(r'\\"\s*]', '"]', response)  # Fix: \" ] -> " ]
        
        # Fix pattern: "text", "other_key": should be "text", \n "other_key":
        # This happens when the LLM forgets to escape quotes properly
        response = re.sub(r'([^\\])"(\s*,\s*)"([^"]+)"(\s*:)', r'\1"\2\n"\3"\4', response)
        
        # CRITICAL FIX: Remove blank lines between JSON properties
        # The LLM adds blank lines for readability but JSON doesn't allow them
        # Pattern: },\n\n"key": -> },\n"key":
        response = re.sub(r',\n\n+', ',\n', response)  # Remove multiple newlines after commas
        response = re.sub(r'{\n\n+', '{\n', response)  # Remove blank lines after opening braces
        response = re.sub(r'\n\n+}', '\n}', response)  # Remove blank lines before closing braces
        response = re.sub(r'\[\n\n+', '[\n', response)  # Remove blank lines after opening brackets
        response = re.sub(r'\n\n+]', '\n]', response)  # Remove blank lines before closing brackets
        
        # Try to extract JSON if there's extra text
        if not response.startswith('{'):
            # Find first { and last }
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1 and end > start:
                response = response[start:end+1]
        
        # PRE-PROCESS: Fix broken arrays BEFORE parsing  
        # Use simple string replacement - much more reliable than regex
        import re
        
        # NEW: Ultra-aggressive fix for the common pattern: ["item"], "item2", "item3"
        # This fixes the exact error we're seeing
        array_fields = ['strengths', 'weaknesses', 'missing_points', 'improvement_suggestions', 'follow_up_questions']
        all_field_names = array_fields + ['correctness', 'completeness', 'clarity', 'confidence', 'technical_depth', 'performance_summary', 'detailed_feedback', 'rating_category']
        
        # First pass: Fix the most common pattern where items are outside the array
        for field in array_fields:
            # Pattern: "field": [...], "orphan1", "orphan2", "next_field":
            # We need to move orphan1, orphan2 inside the array
            pattern = rf'("{field}"\s*:\s*\[[^\]]*\])\s*((?:,\s*"[^"]*")+)\s*,\s*("(?:{"|".join(all_field_names)})")'
            
            def fix_orphans(match):
                array_part = match.group(1)  # "field": [...]
                orphans = match.group(2)      # , "orphan1", "orphan2"
                next_field = match.group(3)   # "next_field"
                
                # Remove the closing bracket from array
                array_without_close = array_part.rstrip(']').rstrip()
                # Add orphans inside the array
                fixed_array = array_without_close + orphans + ']'
                # Add comma before next field
                return fixed_array + ', ' + next_field
            
            response = re.sub(pattern, fix_orphans, response)
        
        # Fix pattern: ],"orphan" -> ,"orphan"]
        # Do this repeatedly for each array field
        max_fixes = 20
        fixes_made = 0
        
        for _ in range(max_fixes):  # Max 20 iterations to prevent infinite loop
            original = response
            
            for field in array_fields:
                # Look for pattern: "field":[...],orphan
                # Find the array closing ] and check if there's an orphaned string after it
                
                for next_field in all_field_names:
                    # Pattern: ],"orphan_text","next_field":
                    # We want to move "orphan_text" before the ]
                    
                    old_pattern = f'],"{next_field}":'
                    # Look backwards from this pattern to find orphaned strings
                    
                    if old_pattern in response:
                        # Find all occurrences
                        pos = 0
                        while True:
                            pos = response.find(old_pattern, pos)
                            if pos == -1:
                                break
                            
                            # Look backwards for a quoted string
                            before = response[:pos]
                            # Find the last quoted string before this position
                            match = re.search(r'],\s*("(?:[^"]|\\")*")\s*,\s*$', before)
                            
                            if match:
                                orphan_text = match.group(1)
                                # Check if this orphan comes after one of our array fields
                                field_check = re.search(f'("{"|".join(array_fields)}"\\s*:\\s*\\[[^\\]]*){re.escape(orphan_text)}', before)
                                
                                if field_check:
                                    # Move the orphan inside the array
                                    # Replace: ...],[orphan],"next -> ...,orphan],"next
                                    response = before[:match.start()] + ',' + orphan_text + '],"' + next_field + '":' + response[pos+len(old_pattern):]
                                    fixes_made += 1
                                    break
                            
                            pos += 1
            
            # If no changes were made, we're done
            if response == original:
                break
        
        if fixes_made > 0:
            logger.debug(f"Fixed {fixes_made} orphaned array strings")
        
        # Also fix plain strings for array fields: "field": "value" -> "field": ["value"]
        for field in array_fields:
            pattern = f'("{field}"\\s*:\\s*)"([^"\\[]+)"'
            if not re.search(f'"{field}"\\s*:\\s*\\[', response):
                response = re.sub(pattern, r'\1["\2"]', response, count=1)
        
        try:
            data = json.loads(response)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation JSON: {e}")
            logger.error(f"Problematic response: {response[:1000]}")
            
            # Try to recover from malformed JSON
            data = None
            
            # Strategy 0: Fix broken arrays more aggressively
            if 'Expecting' in str(e) and data is None:
                try:
                    # Manually find and fix broken arrays
                    fixed_response = response
                    for field in array_fields:
                        # Find each occurrence of the field
                        field_pattern = f'"{field}"\\s*:\\s*\\['
                        matches = list(re.finditer(field_pattern, fixed_response))
                        
                        for match in reversed(matches):  # Process from end to avoid offset issues
                            start = match.end()
                            # Find the matching closing bracket or next field
                            depth = 1
                            i = start
                            in_string = False
                            array_end = -1
                            
                            while i < len(fixed_response) and depth > 0:
                                char = fixed_response[i]
                                if char == '"' and (i == 0 or fixed_response[i-1] != '\\\\'):
                                    in_string = not in_string
                                elif not in_string:
                                    if char == '[':
                                        depth += 1
                                    elif char == ']':
                                        depth -= 1
                                        if depth == 0:
                                            array_end = i
                                i += 1
                            
                            if array_end > 0:
                                # Check if there are orphaned strings after the ]
                                after_array = fixed_response[array_end+1:array_end+100]
                                orphan_pattern = r'^\\s*,\\s*"([^"]*)"'
                                orphans = re.findall(orphan_pattern, after_array)
                                
                                if orphans:
                                    # Insert orphaned strings back into the array
                                    array_content = fixed_response[start:array_end]
                                    for orphan in orphans:
                                        if array_content.strip():
                                            array_content += f', "{orphan}"'
                                        else:
                                            array_content = f'"{orphan}"'
                                    
                                    # Remove orphaned strings from after the array
                                    after_cleaned = re.sub(r'^(\\s*,\\s*"[^"]*")+', '', after_array)
                                    
                                    # Reconstruct
                                    fixed_response = (
                                        fixed_response[:start] +
                                        array_content +
                                        ']' +
                                        after_cleaned +
                                        fixed_response[array_end+len(after_array)+1:]
                                    )
                    
                    data = json.loads(fixed_response)
                    logger.warning("Recovered from broken arrays using aggressive repair")
                except Exception as repair_err:
                    logger.debug(f"Aggressive array repair failed: {repair_err}")
            
            # Strategy 1: If unterminated string, truncate to last complete field
            if 'Unterminated string' in str(e) or ('Expecting' in str(e) and data is None):
                lines = response.split('\n')
                # Try progressively removing lines from the end
                for i in range(len(lines)-1, max(0, len(lines)-15), -1):
                    truncated = '\n'.join(lines[:i])
                    # Remove trailing comma and incomplete content
                    truncated = truncated.rstrip()
                    if truncated.endswith(','):
                        truncated = truncated[:-1]
                    # Remove incomplete string if present
                    if truncated.count('"') % 2 != 0:
                        # Odd number of quotes - remove last incomplete string
                        last_quote = truncated.rfind('"')
                        if last_quote > 0:
                            # Find the field name before this quote
                            before_quote = truncated[:last_quote]
                            last_colon = before_quote.rfind(':')
                            if last_colon > 0:
                                truncated = truncated[:last_colon-1].rstrip(',') if last_colon > 0 else truncated[:last_quote]
                    # Add closing brace if missing
                    if truncated.count('{') > truncated.count('}'):
                        truncated = truncated.rstrip(',') + '\n}'
                    try:
                        data = json.loads(truncated)
                        logger.warning(f"Recovered from truncated JSON by removing {len(lines)-i} lines")
                        break
                    except:
                        continue
            
            # Strategy 2: Try to repair common issues
            if data is None:
                try:
                    import re
                    repaired = response
                    # Add closing brace if missing
                    if repaired.count('{') > repaired.count('}'):
                        repaired = repaired + '}'
                    data = json.loads(repaired)
                    logger.warning("Recovered from JSON by adding closing brace")
                except:
                    pass
            
            # Strategy 3: Fix string values that should be arrays
            if data is None:
                try:
                    import re
                    # Convert string values to arrays for list fields
                    # Pattern: "field":"value" -> "field":["value"]
                    array_fields = ['strengths', 'weaknesses', 'missing_points', 'improvement_suggestions', 'follow_up_questions']
                    repaired = response
                    for field in array_fields:
                        # Match "field":"value" or "field":"value1","value2" patterns
                        pattern = f'("{field}"\s*:\s*)"([^"]+)"'
                        repaired = re.sub(pattern, r'\1["\2"]', repaired)
                    data = json.loads(repaired)
                    logger.warning("Recovered from JSON by converting strings to arrays")
                except Exception as repair_error:
                    logger.debug(f"Array repair failed: {repair_error}")
                    pass
            
            # If still failed, fall back to basic evaluation
            if data is None:
                logger.error(f"Could not recover from JSON error, using basic evaluation")
                return self._basic_evaluation(question, answer)
        
        # Build EvaluationResult from parsed data
        try:
            # Helper function to ensure list fields are lists
            def ensure_list(value):
                if isinstance(value, list):
                    return value
                elif isinstance(value, str):
                    # Split by common delimiters if it looks like multiple items
                    if ',' in value:
                        return [item.strip() for item in value.split(',')]
                    return [value] if value else []
                return []
            
            criteria = EvaluationCriteria(
                correctness=float(data.get("correctness", 5.0)),
                completeness=float(data.get("completeness", 5.0)),
                clarity=float(data.get("clarity", 5.0)),
                confidence=float(data.get("confidence", 5.0)),
                technical_depth=float(data.get("technical_depth", 5.0))
            )
            
            # Calculate overall score (average of criteria)
            overall_score = (
                criteria.correctness +
                criteria.completeness +
                criteria.clarity +
                criteria.confidence +
                criteria.technical_depth
            ) / 5.0
            
            # Parse enhanced feedback fields
            detailed_strengths = []
            if "detailed_strengths" in data and isinstance(data["detailed_strengths"], list):
                from app.services.mock_interview_service import DetailedFeedbackItem
                for item in data["detailed_strengths"]:
                    if isinstance(item, dict):
                        try:
                            detailed_strengths.append(DetailedFeedbackItem(**item))
                        except:
                            pass
            
            detailed_weaknesses = []
            if "detailed_weaknesses" in data and isinstance(data["detailed_weaknesses"], list):
                from app.services.mock_interview_service import DetailedFeedbackItem
                for item in data["detailed_weaknesses"]:
                    if isinstance(item, dict):
                        try:
                            detailed_weaknesses.append(DetailedFeedbackItem(**item))
                        except:
                            pass
            
            answer_comparisons = []
            if "answer_comparisons" in data and isinstance(data["answer_comparisons"], list):
                from app.services.mock_interview_service import AnswerComparison
                for item in data["answer_comparisons"]:
                    if isinstance(item, dict):
                        try:
                            answer_comparisons.append(AnswerComparison(**item))
                        except:
                            pass
            
            return EvaluationResult(
                overall_score=round(overall_score, 1),
                criteria_scores=criteria,
                strengths=ensure_list(data.get("strengths", [])),
                weaknesses=ensure_list(data.get("weaknesses", [])),
                missing_points=ensure_list(data.get("missing_points", [])),
                improvement_suggestions=ensure_list(data.get("improvement_suggestions", [])),
                detailed_strengths=detailed_strengths,
                detailed_weaknesses=detailed_weaknesses,
                answer_comparisons=answer_comparisons,
                performance_summary=data.get("performance_summary", ""),
                detailed_feedback=data.get("detailed_feedback", ""),
                rating_category=data.get("rating_category", "Fair"),
                follow_up_questions=ensure_list(data.get("follow_up_questions", [])),
                model_answer=data.get("model_answer", ""),
                recommended_resources=ensure_list(data.get("recommended_resources", [])),
                key_takeaways=ensure_list(data.get("key_takeaways", []))
            )
        
        except Exception as e:
            logger.error(f"Failed to build EvaluationResult from data: {e}")
            logger.error(f"Data: {data}")
            return self._basic_evaluation(question, answer)
    
    def _basic_evaluation(
        self,
        question: InterviewQuestion,
        answer: UserAnswer
    ) -> EvaluationResult:
        """Fallback evaluation when LLM fails"""
        
        logger.info("Using basic evaluation fallback")
        
        # Simple heuristic based on answer length
        answer_length = len(answer.answer_text.split())
        has_code = bool(answer.code_solution)
        
        # Score based on length and code presence
        base_score = min(10, max(3, answer_length / 20))
        if has_code:
            base_score = min(10, base_score + 2)
        
        criteria = EvaluationCriteria(
            correctness=base_score,
            completeness=base_score - 0.5,
            clarity=base_score,
            confidence=base_score - 1,
            technical_depth=base_score - 1.5
        )
        
        # Generate detailed feedback based on answer characteristics
        performance_summary = (
            f"Your answer contains approximately {answer_length} words"
            f"{' and includes code implementation' if has_code else ''}. "
            f"This evaluation is based on automatic analysis. "
            f"For more detailed AI-powered feedback on technical accuracy, completeness, "
            f"and depth, please ensure the evaluation service is properly configured."
        )
        
        detailed_feedback = (
            f"Based on automatic analysis of your response to the {question.difficulty} level "
            f"{question.interview_type} question about {question.topic}, your answer demonstrates "
            f"{'good engagement with' if answer_length > 50 else 'basic coverage of'} the topic. "
            f"{'Your code solution shows practical application. ' if has_code else ''}"
            f"To receive comprehensive feedback including assessment of technical correctness, "
            f"identification of missing key points, specific improvement suggestions, and follow-up "
            f"questions, the AI evaluation service needs to be available. "
            f"{'Consider elaborating more on your approach and reasoning.' if answer_length < 50 else ''}"
        )
        
        result = EvaluationResult(
            overall_score=round(base_score, 1),
            criteria_scores=criteria,
            strengths=["Provided a response to the question", "Engaged with the interview process"],
            weaknesses=["Detailed AI evaluation currently unavailable", "Manual review recommended for accuracy"],
            missing_points=["Comprehensive technical analysis pending"],
            improvement_suggestions=[
                "Retry submission to get AI-powered detailed feedback",
                "Ensure response covers key aspects of the question",
                "Include examples or code where applicable"
            ],
            performance_summary=performance_summary,
            detailed_feedback=detailed_feedback,
            rating_category="Fair",
            follow_up_questions=[]
        )
        
        logger.info(f"Basic evaluation result: score={result.overall_score}, category={result.rating_category}")
        
        return result
    
    async def _generate_session_questions(
        self,
        interview_type: InterviewType,
        difficulty: DifficultyLevel,
        num_questions: int,
        topic: Optional[str]
    ) -> List[InterviewQuestion]:
        """Generate questions for interview session"""
        
        # Build search query based on interview type
        if interview_type == InterviewType.CODING:
            # For coding, explicitly request coding questions
            query = f"{topic or 'coding'} algorithm coding questions with solutions"
        elif interview_type == InterviewType.TECHNICAL:
            # For technical, explicitly exclude coding and request conceptual questions
            query = f"{topic or 'technical'} conceptual interview questions explain how why when"
        elif interview_type == InterviewType.BEHAVIORAL:
            query = f"behavioral interview questions STAR method {topic or ''}"
        else:
            query = f"{interview_type.value} interview questions"
            if topic:
                query = f"{topic} {query}"
        
        # Get questions from interview intelligence service
        try:
            results = await self.interview_service.search_questions(
                query=query,
                limit=num_questions * 2,  # Get more to filter from
                force_refresh=False
            )
            
            # Convert to InterviewQuestion objects
            questions = []
            for idx, result in enumerate(results):
                # CRITICAL FIX: Filter out coding questions from technical interviews
                is_coding_question = (
                    result.get("is_coding_question", False) or
                    result.get("question_type") == "coding" or
                    result.get("code_solution") is not None
                )
                
                # Skip coding questions if this is a technical (non-coding) interview
                if interview_type == InterviewType.TECHNICAL and is_coding_question:
                    logger.debug(f"Skipping coding question in technical interview: {result.get('question', '')[:50]}")
                    continue
                
                # Skip non-coding questions if this is a coding interview
                if interview_type == InterviewType.CODING and not is_coding_question:
                    logger.debug(f"Skipping non-coding question in coding interview: {result.get('question', '')[:50]}")
                    continue
                
                question = InterviewQuestion(
                    question_id=f"q_{idx}_{uuid.uuid4().hex[:8]}",
                    question_text=result.get("question", ""),
                    interview_type=interview_type,
                    difficulty=difficulty,
                    topic=result.get("topic", topic or "general"),
                    expected_points=result.get("key_concepts", []),
                    sample_answer=result.get("answer")
                )
                questions.append(question)
                
                # Stop when we have enough questions
                if len(questions) >= num_questions:
                    break
            
            logger.info(f"Generated {len(questions)} {interview_type.value} questions (filtered from {len(results)} total)")
            
            # If we didn't get enough questions, use fallback
            if len(questions) < num_questions:
                logger.warning(f"Only got {len(questions)} questions, padding with samples")
                fallback = self._get_sample_questions(interview_type, difficulty, num_questions - len(questions))
                questions.extend(fallback)
            
            return questions[:num_questions]
        
        except Exception as e:
            logger.error(f"Failed to generate questions: {e}")
            # Fallback: return sample questions
            return self._get_sample_questions(interview_type, difficulty, num_questions)
    
    def _get_sample_questions(
        self,
        interview_type: InterviewType,
        difficulty: DifficultyLevel,
        num_questions: int
    ) -> List[InterviewQuestion]:
        """Fallback sample questions"""
        
        samples = {
            InterviewType.CODING: [
                "Implement a function to reverse a string",
                "Find the two numbers that sum to a target in an array",
                "Implement a binary search algorithm"
            ],
            InterviewType.BEHAVIORAL: [
                "Tell me about a time you faced a difficult challenge at work",
                "Describe a situation where you had to work with a difficult team member",
                "Give an example of when you showed leadership"
            ],
            InterviewType.TECHNICAL: [
                "Explain the difference between processes and threads",
                "What is a REST API and how does it work?",
                "Explain database indexing and when to use it"
            ]
        }
        
        question_texts = samples.get(interview_type, samples[InterviewType.TECHNICAL])
        
        questions = []
        for idx, text in enumerate(question_texts[:num_questions]):
            questions.append(InterviewQuestion(
                question_id=f"sample_{idx}",
                question_text=text,
                interview_type=interview_type,
                difficulty=difficulty,
                topic="general",
                expected_points=[]
            ))
        
        return questions
    
    async def _complete_session(self, session: InterviewSession):
        """Complete interview session and calculate final stats"""
        session.completed_at = datetime.utcnow()
        
        # Calculate average score
        if session.evaluations:
            total_score = sum(e.overall_score for e in session.evaluations)
            session.average_score = round(total_score / len(session.evaluations), 1)
        
        logger.info(f"Completed session {session.session_id}, avg score: {session.average_score}")
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get complete session summary with all evaluations"""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Calculate total time spent
        total_time = sum(session.time_per_question.values())
        
        # Calculate total hints used
        total_hints = sum(session.hints_used.values())
        
        return {
            "session_id": session.session_id,
            "started_at": session.started_at.isoformat(),
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "interview_type": session.interview_type,
            "difficulty": session.difficulty,
            "total_questions": session.total_questions,
            "questions_answered": len(session.answers),
            "average_score": session.average_score,
            "total_time_seconds": total_time,
            "total_hints_used": total_hints,
            "time_per_question": session.time_per_question,
            "hints_per_question": session.hints_used,
            "evaluations": [
                {
                    "question_number": i + 1,
                    "question": session.questions[i].question_text,
                    "question_type": session.questions[i].interview_type,
                    "difficulty": session.questions[i].difficulty,
                    "score": eval.overall_score,
                    "rating": eval.rating_category,
                    "summary": eval.performance_summary,
                    "detailed_feedback": eval.detailed_feedback,
                    "strengths": eval.strengths,
                    "weaknesses": eval.weaknesses,
                    "improvement_suggestions": eval.improvement_suggestions,
                    "time_spent_seconds": session.time_per_question.get(session.questions[i].question_id, 0),
                    "hints_used": session.hints_used.get(session.questions[i].question_id, 0),
                    "criteria_scores": {
                        "correctness": eval.criteria_scores.correctness,
                        "completeness": eval.criteria_scores.completeness,
                        "clarity": eval.criteria_scores.clarity,
                        "confidence": eval.criteria_scores.confidence,
                        "technical_depth": eval.criteria_scores.technical_depth
                    }
                }
                for i, eval in enumerate(session.evaluations)
            ],
            "performance_insights": {
                "strongest_area": self._get_strongest_criterion(session),
                "weakest_area": self._get_weakest_criterion(session),
                "consistency": self._calculate_consistency(session),
                "efficiency": "Good" if total_hints <= session.total_questions else "Needs Improvement"
            }
        }
    
    def _get_strongest_criterion(self, session: InterviewSession) -> str:
        """Identify the strongest performance criterion"""
        if not session.evaluations:
            return "N/A"
        
        criteria_avgs = {
            "correctness": sum(e.criteria_scores.correctness for e in session.evaluations) / len(session.evaluations),
            "completeness": sum(e.criteria_scores.completeness for e in session.evaluations) / len(session.evaluations),
            "clarity": sum(e.criteria_scores.clarity for e in session.evaluations) / len(session.evaluations),
            "confidence": sum(e.criteria_scores.confidence for e in session.evaluations) / len(session.evaluations),
            "technical_depth": sum(e.criteria_scores.technical_depth for e in session.evaluations) / len(session.evaluations)
        }
        
        return max(criteria_avgs, key=criteria_avgs.get)
    
    def _get_weakest_criterion(self, session: InterviewSession) -> str:
        """Identify the weakest performance criterion"""
        if not session.evaluations:
            return "N/A"
        
        criteria_avgs = {
            "correctness": sum(e.criteria_scores.correctness for e in session.evaluations) / len(session.evaluations),
            "completeness": sum(e.criteria_scores.completeness for e in session.evaluations) / len(session.evaluations),
            "clarity": sum(e.criteria_scores.clarity for e in session.evaluations) / len(session.evaluations),
            "confidence": sum(e.criteria_scores.confidence for e in session.evaluations) / len(session.evaluations),
            "technical_depth": sum(e.criteria_scores.technical_depth for e in session.evaluations) / len(session.evaluations)
        }
        
        return min(criteria_avgs, key=criteria_avgs.get)
    
    def _calculate_consistency(self, session: InterviewSession) -> str:
        """Calculate performance consistency"""
        if len(session.evaluations) < 2:
            return "N/A"
        
        scores = [e.overall_score for e in session.evaluations]
        variance = sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores)
        std_dev = variance ** 0.5
        
        if std_dev < 1.5:
            return "Very Consistent"
        elif std_dev < 2.5:
            return "Consistent"
        else:
            return "Variable"
    
    async def end_session(self, session_id: str) -> Dict[str, Any]:
        """End session and get final summary"""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if not session.completed_at:
            await self._complete_session(session)
        
        summary = await self.get_session_summary(session_id)
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        return summary


# Global service instance (initialized in main.py)
mock_interview_service: Optional[MockInterviewService] = None


def initialize_mock_interview_service(llm_service, interview_intelligence_service):
    """Initialize the global mock interview service (always use Groq LLM)"""
    from app.services.llm_service import get_llm_service
    global mock_interview_service
    mock_interview_service = MockInterviewService(
        llm_service=get_llm_service("groq"),
        interview_intelligence_service=interview_intelligence_service
    )
    logger.info("Mock Interview Service initialized (Groq LLM)")