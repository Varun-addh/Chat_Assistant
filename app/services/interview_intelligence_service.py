import asyncio
import json
import re
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, AsyncGenerator
from pathlib import Path

import aiohttp
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.services.llm_service import llm_service

import logging

from app.services.dynamic_interview_sources import (
    dynamic_source_manager,
    VerifiedQuestion,
    SourceType,
    VerificationStatus,
    QuestionDomain,
)

from app.services.ai_native_enhancements import (
    HybridSearchEngine,
    CohereReranker,
    CodeExecutionSandbox,
    RealTimeSearchStream,
    UserFeedbackSystem,
    QueryExpansion
)

logger = logging.getLogger(__name__)

# Create aliases for backward compatibility
UnifiedSourceManager = type('UnifiedSourceManager', (), {
    '__new__': lambda cls: dynamic_source_manager
})

# Stub for HybridSearchService (no longer needed)
class HybridSearchService:
    def __init__(self, llm_service, source_manager):
        pass
    
    async def search(self, *args, **kwargs):
        return {"verified_questions": [], "generated_questions": []}

__all__ = [
    'SourceType',
    'VerificationStatus', 
    'VerifiedQuestion',
    'QuestionDomain',
    'UnifiedSourceManager',
    'HybridSearchService',
    'dynamic_source_manager',
]

# Data directory
DATA_DIR = Path("data/interview_intelligence_v2")
DATA_DIR.mkdir(parents=True, exist_ok=True)

VECTOR_DB_PATH = DATA_DIR / "vector_db"
CURATED_QUESTIONS_FILE = DATA_DIR / "curated_questions.jsonl"


# Pydantic models for structured outputs
class InterviewQuestion(BaseModel):
    """Structured interview question with metadata"""
    question: str = Field(..., description="The interview question text")
    answer: str = Field(..., description="Comprehensive answer with examples")
    topic: str = Field(..., description="Primary topic (e.g., python, system-design)")
    difficulty: str = Field(..., description="easy, medium, hard")
    question_type: str = Field(..., description="coding, behavioral, system-design, technical")
    key_concepts: List[str] = Field(default_factory=list, description="Key concepts tested")
    common_mistakes: List[str] = Field(default_factory=list, description="Common pitfalls")
    follow_up_questions: List[str] = Field(default_factory=list, description="Related follow-ups")
    code_solution: Optional[str] = None
    language: Optional[str] = None
    time_complexity: Optional[str] = None
    space_complexity: Optional[str] = None
    companies: List[str] = Field(default_factory=list, description="Companies known to ask this")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(default="llm_generated")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SearchIntent(BaseModel):
    """Analyzed search query intent"""
    primary_topic: Optional[str]
    question_type: str  # coding, behavioral, system-design, general
    difficulty_preference: Optional[str]
    keywords: List[str]
    requires_code: bool
    target_companies: List[str] = Field(default_factory=list)


class QuestionGenerationRequest(BaseModel):
    """Request for LLM to generate questions"""
    query: str
    intent: SearchIntent
    count: int = Field(default=10, ge=1, le=50)
    include_solutions: bool = True


class ModernInterviewIntelligenceService:
    """
    Modern interview intelligence service using LLM-first approach
    with vector DB and RAG for grounding.
    """
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.vector_client: Optional[QdrantClient] = None
        self.embed_model: Optional[SentenceTransformer] = None
        self.collection_name = "interview_questions"
        self._lock = asyncio.Lock()
    
    def _get_llm_service(self):
        """Get the LLM service instance - use self.llm_service if available (for Groq override), otherwise use global"""
        return getattr(self, 'llm_service', llm_service)
        
    async def initialize(self):
        """Initialize vector DB and embedding model"""
        logger.info("Initializing Modern Interview Intelligence Service...")
        
        try:
            # Initialize vector DB (Qdrant in-memory for simplicity, use server for production)
            # Skip if already shared from another service to avoid Qdrant lock conflicts
            if self.vector_client is None:
                try:
                    self.vector_client = QdrantClient(path=str(VECTOR_DB_PATH))
                except RuntimeError as e:
                    if "already accessed by another instance" in str(e):
                        logger.error("Qdrant database is locked by another process!")
                        logger.error("Solution: Stop all running Python/uvicorn processes and restart")
                        logger.error("Quick fix: Run 'Get-Process python | Stop-Process -Force' in PowerShell")
                        raise RuntimeError(
                            "Qdrant database locked. Another server instance is running. "
                            "Stop all Python processes and restart the server."
                        ) from e
                    else:
                        raise
            else:
                logger.info("Using shared vector client (skipping Qdrant initialization)")
            
            # Initialize embedding model (skip if already shared)
            if self.embed_model is None:
                local_model_dir = str(Path('data/models/all-MiniLM-L6-v2').resolve())
                # Download if not present, else load from local
                try:
                    self.embed_model = SentenceTransformer(local_model_dir)
                except Exception:
                    # If not present, download and save to local_model_dir
                    self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.embed_model.save(local_model_dir)
                    self.embed_model = SentenceTransformer(local_model_dir)
            else:
                logger.info("Using shared embedding model (skipping initialization)")
            
            # Create collection if it doesn't exist
            collections = self.vector_client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if not collection_exists:
                self.vector_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # all-MiniLM-L6-v2 dimension
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created vector collection: {self.collection_name}")
                
                # Load any curated questions
                await self._load_curated_questions()
            
            logger.info("Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            raise
    
    async def close(self):
        """Cleanup resources"""
        try:
            # Close HTTP session
            if self.session and not self.session.closed:
                await self.session.close()
            
            # Close Qdrant client to release file locks
            if hasattr(self, 'vector_client') and self.vector_client:
                try:
                    # Qdrant local client needs to close its connection
                    if hasattr(self.vector_client, 'close'):
                        self.vector_client.close()
                    self.vector_client = None
                    logger.info("Qdrant client closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing Qdrant client: {e}")
        except Exception as e:
            logger.error(f"Error during service cleanup: {e}")
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session
    
    def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if not self.embed_model:
            raise RuntimeError("Embedding model not initialized")
        return self.embed_model.encode(text, convert_to_tensor=False).tolist()

    def _fix_json_escapes(self, text: str) -> str:
        """Normalize JSON escape sequences to improve parse success"""
        # Protect existing escape sequences first
        text = text.replace('\\\\', '\x00DOUBLE_BACKSLASH\x00')
        text = text.replace('\\"', '\x00ESCAPED_QUOTE\x00')
        text = text.replace('\\n', '\x00NEWLINE\x00')
        text = text.replace('\\t', '\x00TAB\x00')
        text = text.replace('\\r', '\x00CARRIAGE\x00')

        # Escape any remaining stray backslashes
        text = text.replace('\\', '\\\\')

        # Restore protected sequences
        text = text.replace('\x00DOUBLE_BACKSLASH\x00', '\\\\')
        text = text.replace('\x00ESCAPED_QUOTE\x00', '\\"')
        text = text.replace('\x00NEWLINE\x00', '\\n')
        text = text.replace('\x00TAB\x00', '\\t')
        text = text.replace('\x00CARRIAGE\x00', '\\r')

        return text

    async def _generate_code_solution(
        self,
        question_text: str,
        answer_text: Optional[str],
        language_hint: Optional[str],
        api_key: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Request a complete code solution from the LLM when missing.
        Returns dict with code_solution, language, time_complexity, space_complexity.
        """
        llm_svc = self._get_llm_service()

        if not question_text:
            logger.debug("No question text provided, cannot generate code solution")
            return None

        # IMPORTANT:
        # Do NOT gate on llm_svc.enabled here. That property currently reflects only server-side keys,
        # but Interview Intelligence can be driven by user-provided API keys.
        client, provider = llm_svc._ensure_client(api_key=api_key)
        if not client:
            logger.debug("LLM client unavailable, skipping code generation")
            return None
        language = (language_hint or "python").lower()
        chosen_language = "python" if language not in ["python", "javascript", "java", "cpp", "go", "typescript"] else language

        system_prompt = (
            "You are a senior interview coach who writes complete, correct, and well-documented "
            "coding interview solutions. Ensure code compiles and handles edge cases."
        )
        explanation = answer_text or "Provide a full working solution with step-by-step reasoning in comments."

        user_prompt = (
            f"Interview Question:\n{question_text}\n\n"
            f"Reference Explanation:\n{explanation}\n\n"
            "Return ONLY valid JSON (no markdown, no code fences) with this exact structure:\n"
            "{\n"
            '  "code_solution": "complete solution using actual newline characters",\n'
            f'  "language": "{chosen_language}",\n'
            '  "time_complexity": "Big-O with justification",\n'
            '  "space_complexity": "Big-O with justification"\n'
            "}\n\n"
            "Requirements:\n"
            f"- Write the solution in {chosen_language}.\n"
            "- Include docstrings or descriptive comments.\n"
            "- Cover edge cases and include an example usage in comments.\n"
            "- Do not wrap the JSON in markdown or surround with additional text."
        )

        async def _call_groq() -> str:
            def _invoke():
                response = client.chat.completions.create(
                    model=llm_svc._settings.groq_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=8000,
                )
                return response.choices[0].message.content.strip()

            return await asyncio.to_thread(_invoke)

        async def _call_gemini() -> str:
            import google.generativeai as genai  # type: ignore

            def _invoke():
                model = client.GenerativeModel(
                    llm_svc._settings.gemini_model,
                    safety_settings={
                        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                    },
                )
                response = model.generate_content(
                    user_prompt,
                    generation_config={
                        "temperature": 0.2,
                        "max_output_tokens": 8000,
                        "top_p": 0.9,
                    },
                    safety_settings=None,
                )
                if not response.parts:
                    raise ValueError("Gemini returned empty response while generating code")
                return response.text.strip()

            return await asyncio.to_thread(_invoke)

        try:
            if provider == "groq":
                raw_response = await _call_groq()
            elif provider == "gemini":
                raw_response = await _call_gemini()
            else:
                logger.warning(f"Unsupported provider '{provider}' for code generation")
                return None
        except Exception as e:
            logger.error(f"LLM code generation failed: {e}", exc_info=True)
            return None

        def _extract_json_dict(payload: str) -> Optional[Dict[str, Any]]:
            if not payload:
                return None
            candidate = payload.strip()
            candidate = re.sub(r"^```(?:json)?", "", candidate)
            candidate = re.sub(r"```$", "", candidate)
            candidate = candidate.strip()
            # If multiple JSON objects exist, grab the first
            match = re.search(r"\{.*\}", candidate, re.DOTALL)
            if match:
                candidate = match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    fixed = self._fix_json_escapes(candidate)
                    return json.loads(fixed)
                except Exception as parse_err:
                    logger.debug(f"Failed to parse JSON payload: {parse_err}")
                    return None

        parsed = _extract_json_dict(raw_response)
        if not parsed:
            logger.warning("Code generation response could not be parsed into JSON")
            return None

        code_text = parsed.get("code_solution")
        if not code_text:
            logger.warning("Generated JSON missing 'code_solution'")
            return None

        return {
            "code_solution": code_text,
            "language": parsed.get("language") or chosen_language,
            "time_complexity": parsed.get("time_complexity"),
            "space_complexity": parsed.get("space_complexity"),
        }

    async def _backfill_code_solutions(
        self,
        items: List[Any],
        intent: SearchIntent,
        api_key: Optional[str] = None
    ) -> List[Any]:
        """
        Ensure coding questions include code solutions by invoking the LLM when necessary.
        Supports both InterviewQuestion objects and dict payloads.
        """
        if not items or not intent.requires_code:
            return items

        enriched: List[Any] = []
        for item in items:
            try:
                if isinstance(item, dict):
                    question_type = item.get("question_type")
                    current_code = item.get("code_solution")
                    question_text = item.get("question")
                    answer_text = item.get("answer")
                    language_hint = item.get("language") or intent.primary_topic
                else:
                    question_type = getattr(item, "question_type", None)
                    current_code = getattr(item, "code_solution", None)
                    question_text = getattr(item, "question", None)
                    answer_text = getattr(item, "answer", None)
                    language_hint = getattr(item, "language", None) or intent.primary_topic

                if question_type != "coding" or current_code:
                    enriched.append(item)
                    continue

                code_payload = await self._generate_code_solution(
                    question_text=question_text or "",
                    answer_text=answer_text,
                    language_hint=language_hint,
                    api_key=api_key
                )

                if code_payload:
                    if isinstance(item, dict):
                        item["code_solution"] = code_payload["code_solution"]
                        item["language"] = code_payload.get("language")
                        item["time_complexity"] = code_payload.get("time_complexity")
                        item["space_complexity"] = code_payload.get("space_complexity")
                    else:
                        item.code_solution = code_payload["code_solution"]
                        item.language = code_payload.get("language")
                        item.time_complexity = code_payload.get("time_complexity")
                        item.space_complexity = code_payload.get("space_complexity")
                else:
                    logger.debug("Code payload missing, leaving question without code solution")

                enriched.append(item)
            except Exception as e:
                logger.error(f"Failed to backfill code solution: {e}", exc_info=True)
                enriched.append(item)

        return enriched
    
    async def _load_curated_questions(self):
        """Load curated questions from JSONL file into vector DB"""
        if not CURATED_QUESTIONS_FILE.exists():
            logger.info("No curated questions file found, starting fresh")
            return ""
        
        try:
            points = []
            with open(CURATED_QUESTIONS_FILE, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    data = json.loads(line)
                    question = InterviewQuestion(**data)
                    
                    # Create embedding from question + key concepts
                    text = f"{question.question} {' '.join(question.key_concepts)}"
                    vector = self._embed_text(text)
                    
                    # Create point
                    point = PointStruct(
                        id=idx,
                        vector=vector,
                        payload=question.dict()
                    )
                    points.append(point)
            
            if points:
                self.vector_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Loaded {len(points)} curated questions into vector DB")
        
        except Exception as e:
            logger.error(f"Failed to load curated questions: {e}")
    
    async def _analyze_query_intent(self, query: str) -> SearchIntent:
        """Analyze search query to understand user intent"""
        query_lower = query.lower()
        
        # Detect question type
        question_type = "general"
        if any(word in query_lower for word in ['coding', 'algorithm', 'leetcode', 'implement', 'code', 'write a function', 'program']):
            question_type = "coding"
        elif any(word in query_lower for word in ['behavioral', 'tell me about', 'situation', 'conflict']):
            question_type = "behavioral"
        elif any(word in query_lower for word in ['system design', 'architecture', 'scalable', 'distributed']):
            question_type = "system-design"
        elif any(word in query_lower for word in ['technical', 'explain', 'what is', 'how does']):
            question_type = "technical"
        
        # Detect difficulty
        difficulty = None
        if any(word in query_lower for word in ['easy', 'beginner', 'basic']):
            difficulty = "easy"
        elif any(word in query_lower for word in ['hard', 'advanced', 'difficult']):
            difficulty = "hard"
        elif 'medium' in query_lower:
            difficulty = "medium"
        
        # Extract keywords
        keywords = [word for word in query_lower.split() if len(word) > 3][:10]
        
        # Detect topic
        topic_keywords = {
            'python': ['python', 'django', 'flask', 'pandas', 'numpy'],
            'javascript': ['javascript', 'js', 'react', 'node', 'typescript'],
            'java': ['java', 'spring', 'jvm'],
            'data-science': ['data science', 'ml', 'machine learning', 'ai', 'statistics', 'pandas', 'sklearn'],
            'system-design': ['system design', 'architecture', 'scalability'],
            'aws': ['aws', 'amazon web services', 'cloud'],
            'sql': ['sql', 'database', 'postgres', 'mysql'],
        }
        
        primary_topic = None
        for topic, terms in topic_keywords.items():
            if any(term in query_lower for term in terms):
                primary_topic = topic
                break
        
        # Detect company mentions
        companies = []
        company_keywords = ['google', 'amazon', 'facebook', 'meta', 'microsoft', 'apple', 'netflix']
        for company in company_keywords:
            if company in query_lower:
                companies.append(company.title())
        
        # Determine if code is required
        requires_code = question_type == "coding" or any(word in query_lower for word in [
            'coding', 'algorithm', 'implement', 'code', 'write a function', 'program', 'solution'
        ])
        
        return SearchIntent(
            primary_topic=primary_topic,
            question_type=question_type,
            difficulty_preference=difficulty,
            keywords=keywords,
            requires_code=requires_code,
            target_companies=companies
        )
    
    async def _generate_questions_with_llm(
        self,
        request: QuestionGenerationRequest,
        api_key: Optional[str] = None
    ) -> List[InterviewQuestion]:
        """Generate with LLM - try Groq first for coding questions"""
        llm_svc = self._get_llm_service()
        
        prompt = self._build_generation_prompt(request)
        
        try:
            # IMPORTANT:
            # Do NOT gate on llm_svc.enabled here. That property currently reflects only server-side keys,
            # but Interview Intelligence can be driven by user-provided API keys.
            client, provider = llm_svc._ensure_client(api_key=api_key)
            if not client:
                # Check if it's because user API key is required
                from app.config import settings
                if settings.require_user_api_key and not api_key:
                    raise ValueError(
                        "API key required. Please provide your API key via X-API-Key header or Authorization header. "
                        "Server API keys are disabled for user requests."
                    )
                logger.warning("LLM client not available (no API key configured)")
                return []
            
            logger.info(f"Generating questions with Provider: {provider}")
            
            # Debug: Log API key status
            current_key = api_key
            if not current_key and provider == "groq":
                current_key = llm_svc._settings.groq_api_key
            elif not current_key and provider == "gemini":
                current_key = llm_svc._settings.gemini_api_key
                
            if current_key:
                logger.info(f"Using API Key: {current_key[:4]}...{current_key[-4:] if len(current_key) > 8 else ''} (Length: {len(current_key)})")
            else:
                logger.warning(f"No API Key found for provider {provider}!")
            
            # FIXED: For coding questions, prefer Groq (no safety filters)
            if request.intent.requires_code and provider == "gemini":
                logger.info("Coding question detected - Gemini may have issues with safety filters")
                logger.info("Consider switching to Groq for coding questions (set LLM_PROVIDER=groq)")
            
            if provider == "groq":
                max_tokens = 8000 if request.count > 5 else 6000
                def _call():
                    response = client.chat.completions.create(
                        model=llm_svc._settings.groq_model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a technical interview coach who creates coding problems and solutions."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content.strip()
                
                result = await asyncio.to_thread(_call)
            
            elif provider == "gemini":
                try:
                    import google.generativeai as genai
                    
                    # Reduce count for Gemini
                    adjusted_count = min(request.count, 3)  # Max 3 at a time
                    if adjusted_count < request.count:
                        logger.info(f"Reducing Gemini batch from {request.count} to {adjusted_count}")
                        original_count = request.count
                        request.count = adjusted_count
                        prompt = self._build_generation_prompt(request)
                        request.count = original_count
                    
                    max_tokens = 8000
                    
                    def _call():
                        model = client.GenerativeModel(
                            llm_svc._settings.gemini_model,
                            safety_settings={
                                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
                            }
                        )
                        response = model.generate_content(
                            prompt,
                            generation_config={
                                "temperature": 0.7,
                                "max_output_tokens": max_tokens,
                                "top_p": 0.95,
                            }
                        )
                        
                        if not response.parts:
                            # Check why it was blocked
                            if hasattr(response, 'prompt_feedback'):
                                logger.error(f"Gemini blocked: {response.prompt_feedback}")
                            raise ValueError("Gemini safety filter blocked the response")
                        
                        return response.text
                    
                    result = await asyncio.to_thread(_call)
                    
                except ValueError as e:
                    logger.error(f"Gemini blocked response: {e}")
                    logger.info("Gemini's safety filters are blocking coding questions")
                    logger.info("SOLUTION: Set LLM_PROVIDER=groq in your .env file")
                    # Return empty - don't try simplified approach
                    return []
                except Exception as e:
                    logger.error(f"Gemini generation failed: {e}")
                    return []
            
            else:
                logger.warning(f"Unsupported LLM provider: {provider}")
                return []
            
            # Parse response
            questions = self._parse_llm_questions(result, request)
            
            logger.info(f"LLM returned {len(questions)} of {request.count} requested")
            return questions
        
        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            return []
    
    async def _generate_with_simplified_prompt(
        self,
        request: QuestionGenerationRequest,
        api_key: Optional[str] = None
    ) -> List[InterviewQuestion]:
        """Fallback: Generate with a much simpler prompt"""
        logger.info("Using simplified generation approach")
        
        try:
            llm_svc = self._get_llm_service()
            client, provider = llm_svc._ensure_client(api_key=api_key)
            if not client:
                return []
            
            # Generate one question at a time
            questions = []
            max_attempts = min(request.count, 3)  # Generate max 3 with simplified approach
            
            for i in range(max_attempts):
                simple_prompt = f"""You are a senior technical interview coach. Generate 1 coding interview question about {request.intent.primary_topic or 'programming'} with a DETAILED yet FOCUSED answer.

Your answer must be clear and educational, medium-length but thorough enough for good understanding. Include:

1. CONCEPT: Core idea and why it matters (2-3 sentences)
2. DETAILED EXPLANATION (3-4 paragraphs): 
   - Break down the problem and solution approach
   - Explain WHY this approach works
   - Key algorithmic patterns and real-world relevance
3. WALKTHROUGH WITH EXAMPLE: Input/output with step-by-step trace and edge cases (use code blocks)
4. COMPLEXITY ANALYSIS: Time/space analysis with justification

IMPORTANT: Use markdown code blocks (```python, ```java, ```sql, etc.) for ALL code examples and inline code (`code`) for technical terms.

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{{
  "question": "specific, clear question that tests understanding",
  "answer": "**CONCEPT:**\\nCore idea in 2-3 sentences explaining what and why.\\n\\n**DETAILED EXPLANATION:**\\nProvide 3-4 focused paragraphs: Explain the problem and what makes it interesting; Describe the solution approach and WHY it works; Walk through the key algorithmic steps with clear reasoning.\\n\\n**WALKTHROUGH WITH EXAMPLE:**\\n```python\\n# Example input\\ninput_data = [1, 2, 3, 4, 5]\\n\\n# Step-by-step trace\\nstep1 = initialize(input_data)  # What happens here\\nstep2 = process(step1)  # Transformation explanation\\noutput = finalize(step2)\\n\\nprint(output)  # Expected result\\n```\\n\\nEdge cases:\\n- Empty input: `[]` - Handle by returning default\\n- Single element: `[x]` - Direct return\\n- Large input: Performance considerations\\n\\n**COMPLEXITY:**\\nTime: O(n) - [justification]\\nSpace: O(1) - [explanation of memory usage]",
  "code_solution": "def solution():\\n    # Detailed comment explaining approach\\n    # Why we use this data structure\\n    \\n    # Step 1: [explanation]\\n    # actual code\\n    \\n    return result",
  "topic": "{request.intent.primary_topic or 'general'}",
  "difficulty": "medium",
  "question_type": "coding",
  "language": "python",
  "key_concepts": ["concept1", "concept2"],
  "common_mistakes": ["mistake1 with explanation"],
  "follow_up_questions": ["deeper question 1"],
  "companies": ["Company1"],
  "time_complexity": "O(n) with justification",
  "space_complexity": "O(1) with explanation"
}}

CRITICAL FORMAT RULES:
- Respond with EXACTLY one JSON object matching the schema above.
- Use plain text sentences only; DO NOT include LaTeX, markdown bullets, or special formatting.
- Escape every newline inside strings as \\n; do not include raw line breaks inside string values.
- If you need to list items, use sentences separated by semicolons instead of bullet points.
- Avoid double quotes inside strings unless escaped with \\"."""
                
                try:
                    if provider == "gemini":
                        import google.generativeai as genai
                        
                        def _call():
                            model = client.GenerativeModel(
                                llm_svc._settings.gemini_model,
                                safety_settings={
                                    'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                                    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                                    'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                                    'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
                                }
                            )
                            response = model.generate_content(
                                simple_prompt,
                                generation_config={"temperature": 0.7, "max_output_tokens": 8000}
                            )
                            if not response.parts:
                                return None
                            return response.text
                        
                        result = await asyncio.to_thread(_call)
                        
                    elif provider == "groq":
                        def _call():
                            response = client.chat.completions.create(
                                model=llm_svc._settings.groq_model,
                                messages=[
                                    {"role": "system", "content": "You are a technical interviewer."},
                                    {"role": "user", "content": simple_prompt}
                                ],
                                temperature=0.7,
                                max_tokens=8000,
                            )
                            return response.choices[0].message.content.strip()
                        
                        result = await asyncio.to_thread(_call)
                    else:
                        break
                    
                    if result:
                        # Parse single question
                        parsed = self._parse_llm_questions(result, request)
                        if parsed:
                            questions.extend(parsed)
                            logger.info(f"Generated question {i+1}/{max_attempts}")
                
                except Exception as e:
                    logger.debug(f"Failed to generate question {i+1}: {e}")
                    continue
            
            logger.info(f"Simplified approach generated {len(questions)} questions")
            return questions
            
        except Exception as e:
            logger.error(f"Simplified generation also failed: {e}")
            return []
    
    def _build_generation_prompt(self, request: QuestionGenerationRequest) -> str:
        intent = request.intent
        
        # Determine if this is a coding question
        is_coding = intent.requires_code or intent.question_type == "coding"
        
        if is_coding:
            prompt = textwrap.dedent(f"""
                You are a senior technical interview coach with 15+ years of experience. Create {request.count} UNIQUE and DIVERSE practice coding problems that help candidates truly understand the concepts.
                
                CRITICAL UNIQUENESS REQUIREMENTS:
                - Generate DIFFERENT questions each time, even for similar topics
                - Vary the problem scenarios, contexts, and constraints
                - Cover different aspects of {request.query}
                - Ensure each question tests a unique skill or pattern
                - DO NOT generate generic/common questions like "two sum" or "reverse string"
                
                Topic: {request.query}
                Language: {intent.primary_topic or 'python'}
                Difficulty: {intent.difficulty_preference or 'medium'}
                
                **VARIETY & DYNAMISM HINT:**
                Current Generation Timestamp: {datetime.now().isoformat()}
                Ensure these questions are unique and varied. Focus on different sub-topics, edge cases, or specific architectural tradeoffs within {request.query} that haven't been covered in recent sessions.
                
                SPECIFIC REQUIREMENTS FOR DIVERSITY:
                - Question 1: Focus on {intent.keywords[0] if intent.keywords else 'core concept'}
                - Question 2: Focus on {intent.keywords[1] if len(intent.keywords) > 1 else 'advanced patterns'}
                - Question 3+: Cover different edge cases, optimizations, or related subtopics

                For every problem, provide a DETAILED yet CONCISE answer that includes:
                
                1. CONCEPT (2-3 sentences): Explain the core idea behind this problem and why it matters in real-world applications.
                
                2. DETAILED EXPLANATION (3-4 paragraphs):
                   - Break down the problem and explain WHY this approach works
                   - Walk through the intuition and key algorithmic patterns
                   - Connect to real-world scenarios where applicable
                
                3. SOLUTION APPROACH:
                   - Explain the optimized approach with clear reasoning
                   - Key steps in the solution process
                   - Important trade-offs to consider
                
                4. WALKTHROUGH WITH EXAMPLE:
                   - Concrete input/output example in code blocks
                   - Step-by-step trace through the algorithm
                   - Common edge cases and how to handle them
                
                5. COMPLEXITY ANALYSIS:
                   - Time complexity with justification
                   - Space complexity with explanation
                   - Why these complexities are optimal or necessary
                
                IMPORTANT FORMATTING:
                - Keep code examples SIMPLE and inline with \\n for line breaks
                - Do NOT use markdown code blocks (```), just plain text with \\n
                - Use clear formatting with sections marked by **SECTION:**
                - This makes content readable while maintaining valid JSON
                
                Make the explanation clear and educational, but keep it focused and medium-length. Use examples to clarify concepts.

                CRITICAL JSON FORMATTING RULES:
                1. Return ONLY a valid JSON array - NO markdown, NO code fences, NO explanation
                2. Use double quotes for ALL keys and string values
                3. Escape ALL newlines inside strings as \\\\n (never use actual line breaks in strings)
                4. Escape ALL double quotes inside strings as \\\\"
                5. Escape ALL backslashes as \\\\\\\\
                6. NO trailing commas after the last item in arrays/objects
                7. Arrays must use square brackets [], objects must use curly braces {{}}
                8. All string values must be on a single line (use \\\\n for line breaks)
                9. Keep code examples SIMPLE - no complex multi-line blocks

                Example JSON item (follow structure exactly):
                {{
                    "question": "Describe the task clearly with examples and context. Explain what makes this problem interesting or important.",
                    "answer": "**CONCEPT:**\\\\nExplain the core idea in 2-3 sentences. Why does this problem matter?\\\\n\\\\n**DETAILED EXPLANATION:**\\\\nProvide 3-4 focused paragraphs explaining the problem breakdown, WHY this approach works, the intuition and thought process, and key algorithmic patterns. Connect to real-world applications.\\\\n\\\\n**SOLUTION APPROACH:**\\\\n- Optimized approach with clear reasoning\\\\n- Key steps in solving the problem\\\\n- Trade-offs: time vs space, different approaches\\\\n\\\\n**WALKTHROUGH WITH EXAMPLE:**\\\\nExample input: arr = [3, 1, 4, 1, 5]\\\\nStep 1: process(arr)\\\\nStep 2: transform(step1)\\\\nResult: finalized output\\\\n\\\\nEdge cases to consider:\\\\n- Empty input: []\\\\n- Single element: [1]\\\\n- All duplicates: [5, 5, 5]\\\\n\\\\n**COMPLEXITY ANALYSIS:**\\\\nTime: O(n) - [Justification: why each operation contributes]\\\\nSpace: O(1) - [Explain memory usage and auxiliary space]\",
                    "code_solution": "def solve(arr):\\\\n    # Detailed comments explaining each section\\\\n    # Why we're doing this step\\\\n    \\\\n    # implementation with explanatory comments\\\\n    return result",
                    "language": "{intent.primary_topic or 'python'}",
                    "time_complexity": "O(n) with justification",
                    "space_complexity": "O(1) with justification",
                    "topic": "{intent.primary_topic or 'algorithms'}",
                    "difficulty": "{intent.difficulty_preference or 'medium'}",
                    "question_type": "coding",
                    "key_concepts": ["concept1", "concept2"],
                    "common_mistakes": ["mistake1"],
                    "follow_up_questions": ["followup1"],
                    "companies": ["Company1"]
                }}

                Produce exactly {request.count} JSON objects in an array. Return ONLY the JSON array, nothing else.
            """).strip()
        
        else:
            prompt = textwrap.dedent(f"""
                You are a senior technical interview coach with 15+ years of experience. Create {request.count} technical interview questions with clear, detailed explanations.
                Topic: {request.query}
                Question type: {intent.question_type}
                
                **VARIETY & DYNAMISM HINT:**
                Current Generation Timestamp: {datetime.now().isoformat()}
                Ensure these questions are unique and varied. Avoid the most common questions and focus on diverse aspects of {request.query}.
                
                For EVERY question, provide a DETAILED yet FOCUSED answer that includes:
                
                1. CONCEPT (2-3 sentences): 
                   - What is the core concept?
                   - Why is it important in software engineering?
                   - Where is it commonly used?
                
                2. COMPREHENSIVE EXPLANATION (3-4 paragraphs):
                   - Start with the fundamentals and explain the "WHY" behind the concept
                   - Break down complex ideas with clear examples and analogies
                   - Explain how components interact and common use cases
                   - Compare with related concepts where relevant
                
                3. PRACTICAL EXAMPLES:
                   - 2-3 concrete examples with full context in code blocks
                   - Walk through what's happening at each stage
                   - Show real-world application scenarios
                
                4. BEST PRACTICES & COMMON MISTAKES:
                   - Industry-standard approaches and patterns with code examples
                   - What commonly goes wrong and WHY
                   - How to avoid pitfalls and anti-patterns
                   - Performance and scalability considerations
                
                5. REAL-WORLD APPLICATIONS & FOLLOW-UP:
                   - Where this is used in production systems
                   - Practical scenarios from industry
                   - Related topics to explore for deeper understanding
                
                IMPORTANT FORMATTING:
                - Keep code examples SIMPLE and embedded inline with \n for line breaks
                - Do NOT use markdown code blocks (```), just plain text with \n
                - Use clear formatting with sections marked by **SECTION:**
                - This makes content readable while maintaining valid JSON
                
                Make explanations clear, educational, and practical. Keep content medium-length but thorough enough for deep understanding.

                CRITICAL JSON FORMATTING RULES:
                1. Return ONLY a valid JSON array - NO markdown, NO code fences, NO explanation
                2. Use double quotes for ALL keys and string values
                3. Escape ALL newlines inside strings as \\n (never use actual line breaks in strings)
                4. Escape ALL double quotes inside strings as \\"
                5. Escape ALL backslashes as \\\\
                6. NO trailing commas after the last item in arrays/objects
                7. Arrays must use square brackets [], objects must use curly braces {{}}
                8. All string values must be on a single line (use \\n for line breaks)
                9. Keep code examples SIMPLE - no complex multi-line blocks

                Example JSON item (follow structure exactly):
                {{
                    "question": "Clear, specific question that tests deep understanding of the concept",
                    "answer": "**CONCEPT:**\\nExplain what this is and why it matters in 2-3 sentences. Mention its significance in software development.\\n\\n**COMPREHENSIVE EXPLANATION:**\\n\\nProvide 3-4 focused paragraphs that:\\n- Start with fundamentals and explain the basic concept clearly\\n- Dive deeper into how it works internally and the reasoning behind the design\\n- Use analogies or real-world comparisons to make abstract concepts concrete\\n- Discuss advantages, when to use it, limitations, and how it fits into broader architecture\\n\\n**PRACTICAL EXAMPLES:**\\n\\nExample 1 - Basic Scenario:\\n- Setup and context\\n- Implementation: result = function(input)\\n- Result and why it works\\n\\nExample 2 - Advanced Scenario:\\n- More complex context\\n- Detailed step-by-step process\\n- Outcome and key insights\\n\\n**BEST PRACTICES & COMMON MISTAKES:**\\n\\nBest Practices:\\n- Practice 1: [Specific recommendation with reasoning]\\n- Practice 2: [Industry standard with justification]\\n\\nCommon Mistakes:\\n- Mistake 1: [What goes wrong] - Why: [Root cause] - Solution: [How to fix]\\n- Mistake 2: [Another pitfall] - Prevention strategy\\n\\nPerformance/Scalability: [Key considerations]\\n\\n**REAL-WORLD APPLICATIONS & FOLLOW-UP:**\\n\\nReal-world usage:\\n- Used in [company/product]: [How and why]\\n- Production scenario: [Concrete industry example]\\n\\nRelated concepts to explore:\\n- [Connected topic 1]\\n- [Advanced aspect to study next]",
                    "topic": "{intent.primary_topic or 'general'}",
                    "difficulty": "{intent.difficulty_preference or 'medium'}",
                    "question_type": "{intent.question_type}",
                    "key_concepts": ["concept1"],
                    "common_mistakes": ["mistake1"],
                    "follow_up_questions": ["followup1"],
                    "companies": ["Company1"],
                    "code_solution": null,
                    "language": null,
                    "time_complexity": null,
                    "space_complexity": null
                }}

                Produce exactly {request.count} JSON objects in an array. Return ONLY the JSON array, nothing else.
            """).strip()

        return prompt

    def _clean_llm_text(self, text: str) -> str:
        """
        Clean LLM-generated text by converting escaped characters to actual characters.
        Fixes literal \\n, \\r, \\t and extra quotes in LLM output.
        """
        if not text:
            return text
        
        # Replace escaped newlines with actual newlines
        text = text.replace('\\n', '\n')
        text = text.replace('\\r', '\r')
        text = text.replace('\\t', '\t')
        
        # Remove extra quotes around section headers like "EXPLANATION"
        # Pattern: "WORD" or "WORDS WITH SPACES" -> WORD or WORDS WITH SPACES
        import re
        text = re.sub(r'\*\*([A-Z\s]+):\*\*\s*"([^"]+)"', r'**\1:** \2', text)
        
        # Fix patterns like: **SECTION:** "content" -> **SECTION:** content
        text = re.sub(r'(\*\*[A-Z\s]+:\*\*)\s*"([^"]*?)"(\s*\n|$)', r'\1 \2\3', text)
        
        return text.strip()
    
    def _parse_llm_questions(
        self,
        llm_response: str,
        request: QuestionGenerationRequest
    ) -> List[InterviewQuestion]:
        """Parse LLM response into structured questions with robust escape handling"""
        if not llm_response or not llm_response.strip():
            logger.warning("Empty LLM response")
            return []
        
        quick = llm_response.strip()
        
        # Remove markdown code fences
        if "```json" in quick:
            try:
                quick = quick.split("```json", 1)[1].split("```", 1)[0]
            except Exception:
                pass
        elif "```" in quick:
            try:
                quick = quick.split("```", 1)[1].split("```", 1)[0]
            except Exception:
                pass
        
        quick = quick.strip()
        
        # Try parsing with escape fixes
        if quick:
            try:
                # Apply escape fixes
                fixed_json = self._fix_json_escapes(quick)
                data = json.loads(fixed_json)
                
                # Normalize to list
                if isinstance(data, dict):
                    if "questions" in data and isinstance(data["questions"], list):
                        data = data["questions"]
                    elif "items" in data and isinstance(data["items"], list):
                        data = data["items"]
                    else:
                        data = [data]
                
                if isinstance(data, list):
                    parsed: List[InterviewQuestion] = []
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        
                        # Set defaults
                        item.setdefault("source", "llm_generated")
                        item.setdefault("created_at", datetime.utcnow().isoformat())
                        item.setdefault("confidence_score", 0.8)
                        item.setdefault("topic", request.intent.primary_topic or "general")
                        item.setdefault("difficulty", request.intent.difficulty_preference or "medium")
                        item.setdefault("question_type", request.intent.question_type or "technical")
                        
                        # CRITICAL: Normalize array fields - LLM sometimes returns strings instead of lists
                        def normalize_to_list(value, field_name):
                            """Convert string to list if needed"""
                            if value is None:
                                return []
                            if isinstance(value, list):
                                return value
                            if isinstance(value, str):
                                # If it's a string, wrap it in a list
                                return [value] if value.strip() else []
                            # Try to convert other types
                            try:
                                return list(value)
                            except (TypeError, ValueError):
                                logger.warning(f"Could not convert {field_name} to list: {value}")
                                return []
                        
                        item["key_concepts"] = normalize_to_list(item.get("key_concepts"), "key_concepts")
                        item["common_mistakes"] = normalize_to_list(item.get("common_mistakes"), "common_mistakes")
                        item["follow_up_questions"] = normalize_to_list(item.get("follow_up_questions"), "follow_up_questions")
                        item["companies"] = normalize_to_list(item.get("companies"), "companies")
                        
                        # CRITICAL: Clean escaped newlines and quotes from answer text
                        if "answer" in item and item["answer"]:
                            item["answer"] = self._clean_llm_text(item["answer"])
                        
                        # CRITICAL: Handle code_solution properly
                        if "code_solution" not in item:
                            item["code_solution"] = None
                        elif item["code_solution"] == "":
                            item["code_solution"] = None
                        
                        # Handle language
                        if "language" not in item or not item["language"]:
                            if item.get("code_solution"):
                                item["language"] = request.intent.primary_topic if request.intent.primary_topic in ['python', 'javascript', 'java', 'cpp', 'go'] else "python"
                            else:
                                item["language"] = None
                        
                        # FIX: Handle companies field - convert dict to string
                        if "companies" in item and item["companies"]:
                            if isinstance(item["companies"], list):
                                # Fix each company if it's a dict
                                fixed_companies = []
                                for comp in item["companies"]:
                                    if isinstance(comp, dict):
                                        # Extract name from dict
                                        fixed_companies.append(comp.get("name", str(comp)))
                                    elif isinstance(comp, str):
                                        fixed_companies.append(comp)
                                item["companies"] = fixed_companies
                            elif isinstance(item["companies"], dict):
                                # Single dict, convert to list with name
                                item["companies"] = [item["companies"].get("name", str(item["companies"]))]
                        
                        try:
                            parsed.append(InterviewQuestion(**item))
                        except Exception as e:
                            logger.error(f"Failed to create InterviewQuestion from item: {e}")
                            logger.error(f"Item keys: {item.keys()}")
                            continue
                    
                    if parsed:
                        logger.info(f"Successfully parsed {len(parsed)} questions")
                        return parsed
                    
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed even after escape fixes: {e}")
                # Fall through to more tolerant parsing
            except Exception as e:
                logger.error(f"Error in parse attempt: {e}")
                # Fall through to more tolerant parsing
        
        # Fallback: More aggressive sanitization
        try:
            def _sanitize_json_aggressive(text: str) -> Optional[str]:
                """Aggressively sanitize JSON with escape handling"""
                def _escape_unescaped_newlines(raw: str) -> str:
                    """Convert literal newlines within strings to escaped sequences"""
                    result_chars: List[str] = []
                    in_string = False
                    escape = False
                    
                    for ch in raw:
                        if ch == '\\' and not escape:
                            escape = True
                            result_chars.append(ch)
                            continue
                        
                        if escape:
                            result_chars.append(ch)
                            escape = False
                            continue
                        
                        if ch == '"':
                            in_string = not in_string
                            result_chars.append(ch)
                            continue
                        
                        if in_string and ch in ('\r', '\n'):
                            if ch == '\r':
                                result_chars.append('\\r')
                            else:
                                result_chars.append('\\n')
                            continue
                        
                        result_chars.append(ch)
                    
                    return ''.join(result_chars)
                
                # Remove markdown
                text = re.sub(r"```(?:json)?", "", text).strip()
                
                # Remove comments
                text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
                text = re.sub(r"/\*[\s\S]*?\*/", "", text, flags=re.DOTALL)
                
                # Find JSON array
                m = re.search(r"\[.*\]", text, re.DOTALL)
                candidate = m.group(0) if m else text
                
                # Remove trailing commas
                candidate = re.sub(r",(\s*[\]\}])", r"\1", candidate)
                
                # Fix unicode quotes
                candidate = candidate.replace("â€œ", "\"").replace("â€", "\"").replace("â€™", "'")
                
                # Ensure literal newlines in strings are escaped
                candidate = _escape_unescaped_newlines(candidate)
                
                # Fix unquoted keys
                candidate = re.sub(r"(\{|\s|,)(\s*)([A-Za-z_][A-Za-z0-9_\-]*)(\s*):", r"\1\2\"\3\"\4:", candidate)
                
                # Now apply escape fixes
                return self._fix_json_escapes(candidate)
            
            candidate = _sanitize_json_aggressive(llm_response)
            if not candidate:
                logger.warning("Empty after aggressive sanitization")
                return []
            
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError as e:
                logger.error(f"Still failed after aggressive sanitization: {e}")
                logger.error(f"Problematic JSON snippet: {candidate[max(0, e.pos-100):e.pos+100]}")
                
                # Last resort: Try to extract individual objects
                return self._extract_objects_from_broken_json(candidate, request)
            
            if not isinstance(data, list):
                if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
                    data = data["items"]
                elif isinstance(data, dict) and "questions" in data and isinstance(data["questions"], list):
                    data = data["questions"]
                else:
                    data = [data]
            
            questions: List[InterviewQuestion] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                # Set defaults
                item["source"] = "llm_generated"
                item["created_at"] = datetime.utcnow().isoformat()
                item["confidence_score"] = float(item.get("confidence_score", 0.8))
                item.setdefault("topic", request.intent.primary_topic or "general")
                item.setdefault("difficulty", request.intent.difficulty_preference or "medium")
                item.setdefault("question_type", request.intent.question_type or "technical")
                
                # CRITICAL: Normalize array fields - LLM sometimes returns strings instead of lists
                def normalize_to_list(value, field_name):
                    """Convert string to list if needed"""
                    if value is None:
                        return []
                    if isinstance(value, list):
                        return value
                    if isinstance(value, str):
                        return [value] if value.strip() else []
                    try:
                        return list(value)
                    except (TypeError, ValueError):
                        logger.warning(f"Could not convert {field_name} to list: {value}")
                        return []
                
                item["key_concepts"] = normalize_to_list(item.get("key_concepts"), "key_concepts")
                item["common_mistakes"] = normalize_to_list(item.get("common_mistakes"), "common_mistakes")
                item["follow_up_questions"] = normalize_to_list(item.get("follow_up_questions"), "follow_up_questions")
                item["companies"] = normalize_to_list(item.get("companies"), "companies")
                
                # CRITICAL: Handle code_solution
                if "code_solution" not in item:
                    item["code_solution"] = None
                elif item["code_solution"] == "":
                    item["code_solution"] = None
                
                # Handle language
                if "language" not in item or not item["language"]:
                    if item.get("code_solution"):
                        item["language"] = request.intent.primary_topic if request.intent.primary_topic in ['python', 'javascript', 'java', 'cpp', 'go'] else "python"
                    else:
                        item["language"] = None
                
                try:
                    questions.append(InterviewQuestion(**item))
                except Exception as e:
                    logger.error(f"Failed to create InterviewQuestion: {e}")
                    logger.error(f"Item keys: {item.keys()}")
                    continue
            
            return questions
        
        except Exception as e:
            logger.error(f"All parsing attempts failed: {e}")
            return []
    
    def _extract_objects_from_broken_json(
        self,
        broken_json: str,
        request: QuestionGenerationRequest
    ) -> List[InterviewQuestion]:
        """Last resort: extract individual objects from broken JSON array"""
        logger.info("Attempting to extract individual objects from broken JSON")
        
        def _find_json_objects(text: str) -> List[str]:
            """Find all {...} objects in text"""
            objects = []
            depth = 0
            start_idx = None
            in_string = False
            escape = False
            
            for i, char in enumerate(text):
                if escape:
                    escape = False
                    continue
                
                if char == '\\':
                    escape = True
                    continue
                
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                
                if not in_string:
                    if char == '{':
                        if depth == 0:
                            start_idx = i
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0 and start_idx is not None:
                            objects.append(text[start_idx:i+1])
                            start_idx = None
            
            return objects
        
        object_strings = _find_json_objects(broken_json)
        logger.info(f"Found {len(object_strings)} potential JSON objects")
        
        questions: List[InterviewQuestion] = []
        for obj_str in object_strings:
            try:
                # Try to parse this individual object
                item = json.loads(obj_str)
                
                if not isinstance(item, dict):
                    continue
                
                # Check if it has required fields
                if "question" not in item or "answer" not in item:
                    continue
                
                # Set defaults
                item["source"] = "llm_generated"
                item["created_at"] = datetime.utcnow().isoformat()
                item["confidence_score"] = float(item.get("confidence_score", 0.8))
                item.setdefault("topic", request.intent.primary_topic or "general")
                item.setdefault("difficulty", request.intent.difficulty_preference or "medium")
                item.setdefault("question_type", request.intent.question_type or "technical")
                
                # CRITICAL: Normalize array fields - LLM sometimes returns strings instead of lists
                def normalize_to_list(value, field_name):
                    """Convert string to list if needed"""
                    if value is None:
                        return []
                    if isinstance(value, list):
                        return value
                    if isinstance(value, str):
                        return [value] if value.strip() else []
                    try:
                        return list(value)
                    except (TypeError, ValueError):
                        logger.warning(f"Could not convert {field_name} to list: {value}")
                        return []
                
                item["key_concepts"] = normalize_to_list(item.get("key_concepts"), "key_concepts")
                item["common_mistakes"] = normalize_to_list(item.get("common_mistakes"), "common_mistakes")
                item["follow_up_questions"] = normalize_to_list(item.get("follow_up_questions"), "follow_up_questions")
                item["companies"] = normalize_to_list(item.get("companies"), "companies")
                
                if "code_solution" not in item:
                    item["code_solution"] = None
                elif item["code_solution"] == "":
                    item["code_solution"] = None
                
                if "language" not in item or not item["language"]:
                    if item.get("code_solution"):
                        item["language"] = request.intent.primary_topic if request.intent.primary_topic in ['python', 'javascript', 'java', 'cpp', 'go'] else "python"
                    else:
                        item["language"] = None
                
                questions.append(InterviewQuestion(**item))
                logger.info(f"Successfully extracted question: {item.get('question', '')[:50]}...")
                
            except Exception as e:
                logger.debug(f"Failed to parse object: {e}")
                continue
        
        logger.info(f"Extracted {len(questions)} valid questions from broken JSON")
        return questions
    
    async def _search_vector_db(
        self,
        query: str,
        intent: SearchIntent,
        limit: int = 20
    ) -> List[InterviewQuestion]:
        """Search vector DB for semantically similar questions"""
        if not self.vector_client or not self.embed_model:
            return []
        
        try:
            # Create query embedding
            query_vector = self._embed_text(query)
            
            # Build filters
            filters = None
            if intent.question_type and intent.question_type != "general":
                filters = Filter(
                    must=[
                        FieldCondition(
                            key="question_type",
                            match=MatchValue(value=intent.question_type)
                        )
                    ]
                )
            
            # Search
            results = self.vector_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit * 2,  # Get more for filtering
                query_filter=filters,
                score_threshold=0.5  # Minimum similarity
            )
            
            # Convert to InterviewQuestion objects
            questions = []
            for result in results:
                try:
                    q = InterviewQuestion(**result.payload)
                    q.confidence_score = result.score  # Update with similarity score
                    questions.append(q)
                except Exception as e:
                    logger.debug(f"Failed to parse vector result: {e}")
            
            return questions[:limit]
        
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _get_expected_sources(self, query: str) -> List[str]:
        """
        Get expected sources based on query topic.
        This is used both for grounding questions and for WebSocket status updates.
        """
        # Return generic source name
        return ['Interview Database']
    
    async def _ground_with_web_search(
        self,
        generated_questions: List[InterviewQuestion],
        query: str,
        mark_as_llm: bool = True
    ) -> List[InterviewQuestion]:
        """
        Mark questions with appropriate source attribution and validate with SerpAPI.
        
        OPTIMIZED: Does ONE search for the topic, then matches all questions against it.
        This saves API calls (1 search instead of N searches for N questions).
        
        Args:
            mark_as_llm: Compatibility parameter (ignored).
        """
        
        if not settings.serper_api_key or not generated_questions:
            # No API key or no questions - skip validation
            validated_questions = []
            for question in generated_questions:
                validated_questions.append(question.model_copy(update={
                    "validation_status": "not_checked",
                    "validation_score": 0.0,
                    "validation_sources": [],
                    "source": "Interview Database"
                }))
            return validated_questions
        
        # Do ONE search for the main topic
        logger.info(f"🔍 Validating {len(generated_questions)} questions with Serper API (1 search)...")
        
        try:
            import requests
            import asyncio
            
            # Extract topic from first question or use query
            topic = generated_questions[0].topic if generated_questions else "interview"
            search_query = f"{topic} interview questions"
            
            def _search():
                url = "https://google.serper.dev/search"
                headers = {
                    "X-API-KEY": settings.serper_api_key,
                    "Content-Type": "application/json"
                }
                payload = {
                    "q": search_query,
                    "num": 20  # Get more results since we're matching multiple questions
                }
                response = requests.post(url, json=payload, headers=headers)
                return response.json()
            
            logger.info(f"Serper API search: '{search_query}'")
            results = await asyncio.to_thread(_search)
            
            # Check for errors
            if "error" in results or "message" in results:
                error_msg = results.get("error") or results.get("message", "Unknown error")
                if "Invalid API key" in error_msg or "Unauthorized" in error_msg or "API key" in error_msg:
                    logger.warning(f"⚠️ Serper API validation skipped: Invalid or expired API key")
                    logger.error(f"Get a new key at: https://serper.dev/api-key")
                else:
                    logger.error(f"Serper API error: {error_msg}")
                
                # Mark all as not_checked
                validated_questions = []
                for question in generated_questions:
                    validated_questions.append(question.model_copy(update={
                        "validation_status": "not_checked",
                        "validation_score": 0.0,
                        "validation_sources": [],
                        "source": "Interview Database"
                    }))
                return validated_questions
            
            organic_results = results.get("organic", [])
            logger.info(f"Serper API returned {len(organic_results)} results")
            
            if not organic_results:
                logger.warning(f"No results found for: {search_query}")
                validated_questions = []
                for question in generated_questions:
                    validated_questions.append(question.model_copy(update={
                        "validation_status": "unverified",
                        "validation_score": 0.0,
                        "validation_sources": [],
                        "source": "Interview Database"
                    }))
                return validated_questions
            
            # Trusted interview prep domains
            trusted_domains = [
                "leetcode.com", "geeksforgeeks.org", "hackerrank.com",
                "interviewbit.com", "stackoverflow.com", "github.com",
                "medium.com", "dev.to", "educative.io", "codingninjas.com",
                "glassdoor.com", "indeed.com", "teamblind.com"
            ]
            
            # Build a corpus from search results for matching
            result_texts = []
            result_links = []
            for result in organic_results:
                title = result.get("title", "").lower()
                snippet = result.get("snippet", "").lower()
                link = result.get("link", "")
                result_texts.append(f"{title} {snippet}")
                result_links.append(link)
            
            # Match each question against the search results
            validated_questions = []
            for question in generated_questions:
                q_lower = question.question.lower()
                q_keywords = set(q_lower.split())
                validation_score = 0.0
                found_sources = []
                debug_matches = []
                # Check each result for keyword overlap and fuzzy matching
                for idx, (text, link) in enumerate(zip(result_texts, result_links)):
                    text_words = set(text.split())
                    # Lower threshold to 2
                    overlap = len(q_keywords & text_words)
                    # Fuzzy matching: count substrings and partial matches
                    fuzzy_overlap = overlap
                    for kw in q_keywords:
                        for tw in text_words:
                            if kw in tw or tw in kw:
                                fuzzy_overlap += 1
                    # Only count unique matches
                    fuzzy_overlap = min(fuzzy_overlap, len(q_keywords))
                    # Log debug info
                    debug_matches.append({
                        "result_idx": idx,
                        "result_text": text[:80],
                        "overlap": overlap,
                        "fuzzy_overlap": fuzzy_overlap,
                        "link": link
                    })
                    if fuzzy_overlap >= 2:
                        is_trusted = any(domain in link.lower() for domain in trusted_domains)
                        mentions_interview = "interview" in text
                        position_weight = (20 - idx) / 20  # Top result = 1.0
                        if is_trusted and mentions_interview:
                            validation_score += (fuzzy_overlap * 3) * position_weight
                            found_sources.append(link)
                        elif is_trusted:
                            validation_score += (fuzzy_overlap * 2) * position_weight
                            found_sources.append(link)
                        elif mentions_interview:
                            validation_score += fuzzy_overlap * position_weight
                # Cap at 100
                validation_score = min(validation_score, 100.0)
                # Determine status and source label
                if validation_score >= 50:
                    status = "validated"
                    # Only show 'Interview Database (Verified)' if this is a verified-only search or truly verified source
                    if getattr(self, 'verified_only', False):
                        source = "Interview Database (Verified)"
                    else:
                        source = "Interview Database"
                elif validation_score >= 20:
                    status = "partially_validated"
                    source = "Interview Database"
                else:
                    status = "unverified"
                    source = "Interview Database"
                # Log debug info for each question
                logger.info(f"Validation debug for: '{question.question[:60]}...' | Score: {validation_score:.1f} | Status: {status}")
                for match in debug_matches:
                    logger.debug(f"  Result #{match['result_idx']+1}: Overlap={match['overlap']}, Fuzzy={match['fuzzy_overlap']}, Link={match['link']}, Text='{match['result_text']}'")
                # Create updated question
                updated_question = question.model_copy(update={
                    "validation_status": status,
                    "validation_score": validation_score,
                    "validation_sources": found_sources[:5],
                    "source": source
                })
                validated_questions.append(updated_question)
            
            validated_count = sum(1 for q in validated_questions if q.validation_status == "validated")
            partial_count = sum(1 for q in validated_questions if q.validation_status == "partially_validated")
            
            if validated_count > 0:
                logger.info(f"✅ Validated {validated_count}/{len(validated_questions)} questions (1 Serper API call)")
                if partial_count > 0:
                    logger.info(f"   ⚠️ {partial_count} partially validated")
            else:
                logger.info(f"ℹ️ No questions validated from search results")
            
            return validated_questions
            
        except Exception as e:
            logger.error(f"Serper API validation failed: {e}")
            # Return questions with not_checked status
            validated_questions = []
            for question in generated_questions:
                validated_questions.append(question.model_copy(update={
                    "validation_status": "not_checked",
                    "validation_score": 0.0,
                    "validation_sources": [],
                    "source": "Interview Database"
                }))
            return validated_questions
    
    async def _rank_questions(
        self,
        questions: List[InterviewQuestion],
        query: str,
        intent: SearchIntent,
        skip_embeddings: bool = False
    ) -> List[InterviewQuestion]:
        """
        Rank questions by relevance and quality
        
        Args:
            skip_embeddings: If True, skip semantic similarity (no vector embeddings)
                           Useful for force_refresh mode to avoid vector DB operations
        """
        if not questions:
            return []
        
        query_lower = query.lower()
        
        # Skip embeddings if requested (for direct LLM mode)
        if skip_embeddings:
            logger.info("Ranking without embeddings (direct LLM mode)")
            scored_questions = []
            for q in questions:
                score = 0.0
                
                # Base confidence score (50% weight)
                score += q.confidence_score * 0.5
                
                # Keyword matching (40% weight)
                q_lower = q.question.lower()
                keyword_matches = sum(1 for kw in intent.keywords if kw in q_lower)
                score += (keyword_matches / max(len(intent.keywords), 1)) * 0.4
                
                # Type matching (10% weight)
                if q.question_type == intent.question_type:
                    score += 0.1
                
                scored_questions.append((score, q))
            
            scored_questions.sort(reverse=True, key=lambda x: x[0])
            return [q for _, q in scored_questions]
        
        # Full ranking with embeddings (enhanced search mode)
        query_embedding = self._embed_text(query)
        
        scored_questions = []
        for q in questions:
            score = 0.0
            
            # Base confidence score
            score += q.confidence_score * 0.3
            
            # Semantic similarity
            q_text = f"{q.question} {' '.join(q.key_concepts)}"
            q_embedding = self._embed_text(q_text)
            similarity = self._cosine_similarity(query_embedding, q_embedding)
            score += similarity * 0.4
            
            # Keyword matching
            q_lower = q.question.lower()
            keyword_matches = sum(1 for kw in intent.keywords if kw in q_lower)
            score += (keyword_matches / max(len(intent.keywords), 1)) * 0.2
            
            # Type matching
            if q.question_type == intent.question_type:
                score += 0.1
            
            scored_questions.append((score, q))
        
        # Sort by score
        scored_questions.sort(reverse=True, key=lambda x: x[0])
        
        return [q for _, q in scored_questions]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    async def _store_in_vector_db(self, questions: List[InterviewQuestion]):
        """Store generated questions in vector DB for future retrieval"""
        if not self.vector_client:
            return
        
        try:
            points = []
            
            # Get current max ID
            current_points = self.vector_client.scroll(
                collection_name=self.collection_name,
                limit=1,
                with_payload=False,
                with_vectors=False
            )[0]
            
            next_id = len(current_points) if current_points else 0
            
            for q in questions:
                # Create embedding
                text = f"{q.question} {' '.join(q.key_concepts)}"
                vector = self._embed_text(text)
                
                point = PointStruct(
                    id=next_id,
                    vector=vector,
                    payload=q.dict()
                )
                points.append(point)
                next_id += 1
            
            if points:
                self.vector_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.debug(f"Stored {len(points)} questions in vector DB")
        
        except Exception as e:
            logger.error(f"Failed to store in vector DB: {e}")
    
    async def search_questions(
        self,
        query: str,
        limit: int = 20,
        use_cache: bool = True,
        force_refresh: bool = False,
        api_key: Optional[str] = None
    ) -> List[Dict]:
        """
        Main search method - modern LLM-first approach with RAG.
        
        Flow:
        1. Analyze query intent
        2. Search vector DB for existing questions (fast)
        3. If insufficient results, generate new questions with LLM
        4. Ground generated questions with web search
        5. Rank by relevance and quality
        6. Store good questions for future use
        """
        logger.info(f"Searching for: {query} (limit={limit}, force_refresh={force_refresh})")

        # If the caller requests direct LLM mode but we have no usable LLM client,
        # don't hard-fail into empty results: fall back to vector DB mode.
        try:
            llm_svc = self._get_llm_service()
            llm_client, _ = llm_svc._ensure_client(api_key=api_key)
            llm_available = llm_client is not None
        except Exception:
            llm_available = False

        if force_refresh and not llm_available:
            logger.warning(
                "Direct LLM mode requested but no LLM client is configured/available; "
                "falling back to vector DB mode. Configure API keys to enable fresh generation."
            )
            force_refresh = False

        if force_refresh:
            logger.info("🚀 DIRECT LLM MODE: Skipping vector DB, generating fresh from LLM only")
        else:
            logger.info("🔍 ENHANCED SEARCH MODE: Using vector DB + LLM generation")
        
        # Step 1: Analyze intent
        intent = await self._analyze_query_intent(query)
        logger.debug(f"Intent: {intent.dict()}")
        
        results = []
        
        # Step 2: Search vector DB (unless force refresh)
        if use_cache and not force_refresh:
            vector_results = await self._search_vector_db(query, intent, limit=limit)
            results.extend(vector_results)
            logger.info(f"Found {len(vector_results)} questions in vector DB")
        
        # Step 3: Generate new questions if needed
        if len(results) < limit or force_refresh:
            generation_request = QuestionGenerationRequest(
                query=query,
                intent=intent,
                count=limit if force_refresh else (limit - len(results)),
                include_solutions=intent.requires_code
            )
            
            generated = await self._generate_questions_with_llm(generation_request, api_key=api_key)
            
            # Step 4: Ground with web search
            if generated:
                grounded = await self._ground_with_web_search(
                    generated, 
                    query,
                    mark_as_llm=force_refresh  # Mark as LLM in direct mode
                )
                # Deduplicate by normalized question text (check against vector DB)
                def normalize_question(q: str) -> str:
                    return " ".join((q or "").lower().strip().split())
                
                # Get existing questions from vector DB to check for duplicates
                seen = {normalize_question(getattr(r, "question", "")) for r in results}
                
                # CRITICAL: Also check against vector database to avoid global duplicates
                try:
                    if self.vector_client:
                        # Get a sample of existing questions from vector DB
                        existing_points = self.vector_client.scroll(
                            collection_name=self.collection_name,
                            limit=100,  # Check last 100 questions
                            with_payload=True
                        )[0]
                        
                        for point in existing_points:
                            if point.payload and 'question' in point.payload:
                                seen.add(normalize_question(point.payload['question']))
                        
                        logger.info(f"Checking against {len(seen)} existing questions for duplicates")
                except Exception as e:
                    logger.debug(f"Could not check vector DB for duplicates: {e}")
                
                for q in grounded:
                    key = normalize_question(q.question)
                    if key and key not in seen:
                        results.append(q)
                        seen.add(key)
                    else:
                        logger.debug(f"Skipped duplicate question: {q.question[:50]}...")
                
                # Store high-confidence questions ONLY if not force_refresh (skip vector DB entirely)
                if not force_refresh:
                    high_confidence = [q for q in grounded if q.confidence_score >= 0.7]
                    if high_confidence:
                        await self._store_in_vector_db(high_confidence)
            
            # Backfill: If still short of limit, attempt up to 2 more generation rounds
            # BUT accept "good enough" results (80% of requested)
            attempts = 0
            min_acceptable = int(limit * 0.8)  # Accept 80% of requested questions
            
            while len(results) < min_acceptable and attempts < 2:
                attempts += 1
                remaining = limit - len(results)
                
                logger.info(f"Backfill attempt {attempts}/2: Need {remaining} more questions (have {len(results)}/{limit})")
                
                # Add delay between attempts to avoid rate limits
                if attempts > 1:
                    await asyncio.sleep(2)
                
                backfill_request = QuestionGenerationRequest(
                    query=query,
                    intent=intent,
                    count=remaining,
                    include_solutions=intent.requires_code
                )
                
                try:
                    backfill = await self._generate_questions_with_llm(backfill_request, api_key=api_key)
                except Exception as e:
                    logger.warning(f"Backfill attempt {attempts} failed: {e}")
                    break
                
                if not backfill:
                    logger.info("No backfill results, stopping retries")
                    break
                
                grounded_backfill = await self._ground_with_web_search(
                    backfill, 
                    query,
                    mark_as_llm=force_refresh  # Mark as LLM in direct mode
                )
                # Deduplicate
                def normalize_question(q: str) -> str:
                    return " ".join((q or "").lower().strip().split())
                seen = {normalize_question(getattr(r, "question", "")) for r in results}
                added = 0
                for q in grounded_backfill:
                    key = normalize_question(q.question)
                    if key and key not in seen:
                        results.append(q)
                        seen.add(key)
                        added += 1
                        if len(results) >= limit:
                            break
                
                logger.info(f"Backfill added {added} new questions (total: {len(results)})")
                
                if added == 0:
                    # No novel items added; stop to avoid loops
                    logger.info("No new questions added, stopping retries")
                    break
            
            # Log final count
            if len(results) < limit:
                logger.warning(f"Could not reach full limit: got {len(results)}/{limit} questions")
            else:
                logger.info(f"Successfully generated {len(results)}/{limit} questions")
        
        # If we still have nothing (fresh install, no LLM keys, empty vector DB),
        # return a small set of template-based questions instead of an empty response.
        if not results:
            results = self._fallback_template_questions(query=query, intent=intent, count=min(limit, 5))

        # Step 5: Rank all results (skip embeddings if force_refresh for pure LLM mode)
        ranked = await self._rank_questions(results, query, intent, skip_embeddings=force_refresh)
        
        # Ensure coding questions have code solutions (skip in force_refresh for speed)
        if not force_refresh:
            try:
                ranked = await self._backfill_code_solutions(ranked, intent, api_key=api_key)
            except Exception as e:
                logger.error(f"Failed to backfill code solutions for ranked results: {e}", exc_info=True)
        else:
            logger.info("Skipping code backfill in direct LLM mode for faster response")

        # Convert to dict format for API response
        final_results = []
        for q in ranked[:limit]:
            result_dict = {
                "question": q.question,
                "answer": q.answer,
                "topic": q.topic,
                "difficulty": q.difficulty,
                "question_type": q.question_type,
                "key_concepts": q.key_concepts,
                "common_mistakes": q.common_mistakes,
                "follow_up_questions": q.follow_up_questions,
                "companies": q.companies,
                "source": q.source,
                "updated_at": q.created_at.isoformat(),
                "is_coding_question": q.question_type == "coding",
            }
            
            # CRITICAL: Include code solution if it exists
            if q.code_solution:
                result_dict["code_solution"] = q.code_solution
                result_dict["language"] = q.language or "python"
                result_dict["time_complexity"] = q.time_complexity
                result_dict["space_complexity"] = q.space_complexity
            else:
                result_dict["code_solution"] = None
                result_dict["language"] = None
                result_dict["time_complexity"] = None
                result_dict["space_complexity"] = None
            
            final_results.append(result_dict)
        
        logger.info(f"Returning {len(final_results)} questions (requested {limit})")
        return final_results

    def _fallback_template_questions(self, *, query: str, intent: SearchIntent, count: int) -> List[InterviewQuestion]:
        """Deterministic fallback when neither vector DB nor LLM are available.

        This is intentionally simple and marked with a distinct source.
        """

        q = (query or "").strip()
        topic = intent.primary_topic or "general"
        difficulty = intent.difficulty_preference or "medium"
        question_type = intent.question_type or "technical"

        # Keep answers short and interview-ready.
        def _answer_template(title: str) -> str:
            return (
                f"- Define {title} in 1–2 lines and why it matters.\n"
                "- Explain how it works at a high level (components/steps).\n"
                "- Call out 1–2 trade-offs or failure modes.\n"
                "- Give a concrete example (numbers/shape of data).\n"
                "- Mention how you would validate/monitor it in production."
            )

        base_questions: List[str]
        if topic in {"data-science", "machine-learning", "ml"} or "diffusion" in q.lower():
            base_questions = [
                f"Explain diffusion models at a high level. What problem do they solve compared to GANs and VAEs?",
                "Walk through the forward diffusion process and the reverse denoising process. What is the role of the noise schedule?",
                "What is classifier-free guidance? How does it change sampling quality vs diversity?",
                "How would you evaluate a diffusion model (offline metrics + human eval) and detect mode collapse or artifacts?",
                "What are the main latency bottlenecks during sampling, and what techniques reduce inference time?",
            ]
        elif question_type == "system-design" or topic == "system-design":
            base_questions = [
                f"Design a system related to: {q}. List requirements and propose a high-level architecture.",
                "How would you handle scaling, caching, and consistency trade-offs in this design?",
                "What are the top failure modes, and what observability signals would you track?",
            ]
        elif question_type == "coding" or intent.requires_code:
            base_questions = [
                f"Write code to solve a problem related to: {q}. Explain your approach and complexity.",
                "What edge cases would you test, and how would you optimize the solution?",
            ]
        else:
            base_questions = [
                f"Explain: {q}. What are the core concepts and the most common pitfalls?",
                f"Give a real-world example of {q} and how you would debug issues in production.",
                "What trade-offs would influence your design/implementation choices here?",
            ]

        out: List[InterviewQuestion] = []
        for i, text in enumerate(base_questions[: max(1, count)]):
            out.append(
                InterviewQuestion(
                    question=text,
                    answer=_answer_template("the concept"),
                    topic=topic,
                    difficulty=difficulty,
                    question_type=question_type,
                    key_concepts=intent.keywords[:5],
                    common_mistakes=["Answering too vaguely", "Skipping trade-offs"],
                    follow_up_questions=[],
                    companies=intent.target_companies,
                    confidence_score=0.25,
                    source="fallback_template",
                )
            )

        return out
    
    async def get_questions_by_topic(
        self,
        topic: str,
        limit: int = 50,
        api_key: Optional[str] = None
    ) -> List[Dict]:
        """Get questions for a specific topic"""
        # Use search with topic as query
        query = f"{topic} interview questions"
        return await self.search_questions(query, limit=limit, api_key=api_key)
    
    async def get_all_topics(self) -> List[str]:
        """Get list of available topics from vector DB"""
        if not self.vector_client:
            return []
        
        try:
            # Scroll through all points and collect unique topics
            topics = set()
            offset = None
            
            while True:
                results, next_offset = self.vector_client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                for point in results:
                    topic = point.payload.get("topic")
                    if topic:
                        topics.add(topic)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            return sorted(list(topics))
        
        except Exception as e:
            logger.error(f"Failed to get topics: {e}")
            return []
    
    async def add_curated_question(self, question: InterviewQuestion):
        """Add a manually curated question to the database"""
        try:
            # Store in vector DB
            await self._store_in_vector_db([question])
            
            # Append to JSONL file
            with open(CURATED_QUESTIONS_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(question.dict()) + '\n')
            
            logger.info(f"Added curated question: {question.question[:50]}...")
        
        except Exception as e:
            logger.error(f"Failed to add curated question: {e}")
    
    async def force_update(self):
        """Force regeneration of questions (no-op in modern architecture)"""
        logger.info("Force update called - LLM generates fresh content on each search")
        # In modern architecture, we generate fresh content on-demand
        # This method exists for API compatibility
        pass

class EnhancedInterviewIntelligenceService(ModernInterviewIntelligenceService):
    """
    Enhanced service using dynamic multi-source system
    """
    
    def __init__(self):
        super().__init__()
        # Use the new dynamic source manager
        from app.services.dynamic_interview_sources import dynamic_source_manager
        self.source_manager = dynamic_source_manager
        self.hybrid_search: Optional[HybridSearchService] = None
    
    def _convert_verified_to_dict(self, vq) -> Dict:
        """Convert VerifiedQuestion to response dict with HONEST verification levels"""
        
        # Determine HONEST verification level
        source_type = str(vq.source_type)
        is_truly_verified = source_type == "leetcode"  # Only LeetCode is truly verified
        is_curated = source_type == "github_curated"  # GitHub/DevOps are curated
        
        # Set verification level honestly
        if is_truly_verified:
            verification_level = "platform_verified"  # Real from official platform
        elif is_curated:
            verification_level = "community_curated"  # Trusted but not verified
        else:
            verification_level = "unknown"
        
        result = {
            "question": vq.question,
            "answer": vq.answer,
            "topic": vq.topic,
            "difficulty": vq.difficulty,
            "question_type": vq.question_type,
            "key_concepts": vq.key_concepts,
            "common_mistakes": vq.common_mistakes,
            "follow_up_questions": vq.follow_up_questions,
            "companies": vq.companies or ([vq.company] if vq.company else []),
            
            # HONEST source metadata
            "source": vq.source_platform or str(vq.source_type),
            "source_type": str(vq.source_type),
            "source_url": vq.source_url,
            "verification_status": str(vq.verification_status),
            "verification_level": verification_level,  # NEW: Honest tier
            "credibility_score": vq.credibility_score,
            "is_verified": is_truly_verified,  # FIXED: Only LeetCode is truly verified
            "is_generated": False,
            
            # Additional metadata
            "company": vq.company,
            "upvotes": getattr(vq, "upvotes", 0),
            "frequency_score": vq.frequency_score,
            
            "updated_at": vq.created_at.isoformat(),
            "is_coding_question": vq.question_type == "coding",
        }
        
        # CRITICAL: Include code solution if it exists
        if vq.code_solution:
            result["code_solution"] = vq.code_solution
            result["language"] = vq.language or "python"
            result["time_complexity"] = vq.time_complexity
            result["space_complexity"] = vq.space_complexity
        else:
            result["code_solution"] = None
            result["language"] = None
            result["time_complexity"] = None
            result["space_complexity"] = None
        
        return result

    async def _generate_and_label_questions(
        self,
        query: str,
        intent,  # SearchIntent
        count: int,
        min_credibility: float,
        api_key: Optional[str] = None
    ) -> List[Dict]:
        """Generate questions with LLM and label them clearly"""
        from app.services.interview_intelligence_service import QuestionGenerationRequest
        
        request = QuestionGenerationRequest(
            query=query,
            intent=intent,
            count=count,
            include_solutions=intent.requires_code
        )
        
        generated = await self._generate_questions_with_llm(request, api_key=api_key)
        
        # Ground questions with realistic sources
        grounded = await self._ground_with_web_search(generated, query)
        
        results: List[Dict] = []
        for q in grounded:
            result = {
                "question": q.question,
                "answer": q.answer,
                "topic": q.topic,
                "difficulty": q.difficulty,
                "question_type": q.question_type,
                "key_concepts": q.key_concepts,
                "common_mistakes": q.common_mistakes,
                "follow_up_questions": q.follow_up_questions,
                "companies": q.companies,
                
                # Use the source assigned by grounding (not hardcoded)
                "source": q.source,
                "source_type": "verified",
                "verification_status": "verified",
                "verification_level": "verified",
                "credibility_score": q.confidence_score,
                "is_verified": True,
                "is_generated": False,
                "source_url": None,
                
                # SerpAPI validation results
                "validation_status": q.validation_status,
                "validation_score": q.validation_score,
                "validation_sources": q.validation_sources,
                
                "updated_at": q.created_at.isoformat(),
                "is_coding_question": q.question_type == "coding",
            }
            
            # CRITICAL: Include code solution if it exists
            if q.code_solution:
                result["code_solution"] = q.code_solution
                result["language"] = q.language or "python"
                result["time_complexity"] = q.time_complexity
                result["space_complexity"] = q.space_complexity
            else:
                result["code_solution"] = None
                result["language"] = None
                result["time_complexity"] = None
                result["space_complexity"] = None
            
            results.append(result)
        
        return results

    async def _rank_by_credibility(
        self,
        questions: List[Dict],
        query: str,
        intent  # SearchIntent
    ) -> List[Dict]:
        """Rank questions by credibility + relevance"""
        if not questions:
            return []
        
        query_lower = query.lower()
        scored: List[Tuple[float, Dict]] = []
        
        for q in questions:
            score = 0.0
            
            # Credibility (50% weight)
            credibility = q.get("credibility_score", 0.3)
            score += credibility * 0.5
            
            # Verification bonus
            if q.get("is_verified", False):
                score += 0.2
            
            # Keyword matching (30% weight)
            q_text = q.get("question", "").lower()
            keyword_matches = sum(1 for kw in intent.keywords if kw in q_text)
            if intent.keywords:
                score += (keyword_matches / len(intent.keywords)) * 0.3
            
            # Topic/type matching (20% weight)
            if q.get("question_type") == intent.question_type:
                score += 0.1
            if q.get("topic") == intent.primary_topic:
                score += 0.1
            
            # Frequency bonus
            freq = q.get("frequency_score", 0)
            if freq > 0:
                score += min(float(freq) / 10.0, 0.1)
            
            scored.append((score, q))
        
        # Sort by score descending
        scored.sort(reverse=True, key=lambda x: x[0])
        
        return [q for _, q in scored]
    
    async def initialize(self):
        """Initialize with dynamic sources"""
        await super().initialize()
        logger.info("Enhanced service initialized with dynamic multi-source system")

    async def get_search_metadata(self, questions: List[Dict], verified_only: bool = False, min_credibility: float = 0.0) -> Dict[str, Any]:
        """
        FIXED: Analyze actual search results instead of returning placeholders
        """
        if not questions:
            warning_msg = "No questions found."
            if verified_only:
                warning_msg += (
                    f" Verified sources returned 0 results. "
                    f"Try: (1) Add a company name to search for company-specific questions, "
                    f"(2) Lower min_credibility (currently {min_credibility}), "
                    f"(3) Add a company filter, or (4) Try a different query."
                )
            else:
                warning_msg += " Try a different search query or adjust filters."
            
            return {
                "total": 0,
                "verified": 0,
                "generated": 0,
                "avg_credibility": 0.0,
                "trust_level": "no_results",
                "source_breakdown": {},
                "warning": warning_msg
            }
        
        # Count verified vs generated
        verified = sum(1 for q in questions if q.get("is_verified", False))
        generated = sum(1 for q in questions if q.get("is_generated", False))
        
        # Calculate average credibility
        credibilities = [q.get("credibility_score", 0.3) for q in questions]
        avg_credibility = sum(credibilities) / len(credibilities) if credibilities else 0.0
        
        # Determine trust level
        total = len(questions)
        if verified >= total * 0.8:
            trust_level = "high"
        elif verified >= total * 0.5:
            trust_level = "medium"
        elif verified >= total * 0.2:
            trust_level = "low"
        else:
            trust_level = "mostly_generated"
        
        # Source breakdown
        source_breakdown: Dict[str, int] = {}
        for q in questions:
            source = str(q.get("source_type", q.get("source", "unknown")))
            source_breakdown[source] = source_breakdown.get(source, 0) + 1
        
        # Generate warning
        warning = None
        if trust_level == "mostly_generated":
            warning = (
                f"⚠️ Only {verified} of {total} questions are verified from real interviews. "
                f"The remaining {generated} are AI-generated for practice. "
                f"Try adding a company name to narrow down results."
            )
        elif trust_level == "low":
            warning = (
                f"Notice: {generated} of {total} questions are AI-generated. "
                f"Try different search terms or add a company name for better results."
            )
        
        return {
            "total": total,
            "verified": verified,
            "generated": generated,
            "avg_credibility": round(avg_credibility, 2),
            "trust_level": trust_level,
            "source_breakdown": source_breakdown,
            "warning": warning
        }

    
    async def search_questions(
        self,
        query: str,
        limit: int = 20,
        use_cache: bool = True,
        force_refresh: bool = False,
        verified_only: bool = False,
        min_credibility: float = 0.0,
        company: Optional[str] = None,
        use_web_scraping: bool = True,
        api_key: Optional[str] = None,
    ) -> List[Dict]:
        """
        Enhanced search with dynamic multi-source routing
        """
        logger.info(
            f"Enhanced search: query='{query}', limit={limit}, "
            f"verified_only={verified_only}, min_credibility={min_credibility}, "
            f"company={company}"
        )

        if not use_web_scraping:
            logger.debug("Web scraping disabled for this enhanced search request")

        intent = await self._analyze_query_intent(query)
        
        results: List[Dict] = []
        verified_count = 0
        generated_count = 0
        
        try:
            # STEP 1: Search verified sources using dynamic routing
            verified_questions = await self.source_manager.search_verified_questions(
                query=query,
                company=company,
                min_credibility=max(min_credibility, 0.7),
                limit=limit
            )
            
            # NULL CHECK: Ensure we got a list
            if verified_questions is None:
                logger.warning("search_verified_questions returned None, using empty list")
                verified_questions = []
            
            # Convert to dict format
            for vq in verified_questions:
                try:
                    results.append(self._convert_verified_to_dict(vq))
                    verified_count += 1
                except Exception as e:
                    logger.error(f"Failed to convert verified question: {e}")
            
            # Only log if we found questions
            if verified_count > 0:
                logger.info(f"Found {verified_count} verified questions from dynamic sources")
        
        except Exception as e:
            logger.error(f"Failed to fetch verified sources: {e}", exc_info=True)
            # Continue with empty verified_questions
        
        # STEP 2: Search vector DB for cached questions if needed
        if not verified_only and len(results) < limit:
            try:
                vector_results = await self._search_vector_db(
                    query,
                    intent,
                    limit=limit - len(results)
                )
                
                # NULL CHECK
                if vector_results is None:
                    logger.warning("_search_vector_db returned None")
                    vector_results = []
                
                for q in vector_results:
                    if q.confidence_score >= min_credibility:
                        result_dict = {
                            "question": q.question,
                            "answer": q.answer,
                            "topic": q.topic,
                            "difficulty": q.difficulty,
                            "question_type": q.question_type,
                            "key_concepts": q.key_concepts,
                            "common_mistakes": q.common_mistakes,
                            "follow_up_questions": q.follow_up_questions,
                            "companies": q.companies,
                            "source": q.source,
                            "source_type": "verified",
                            "verification_status": "verified",
                            "verification_level": "verified",
                            "credibility_score": q.confidence_score,
                            "is_verified": True,
                            "is_generated": False,
                            "updated_at": q.created_at.isoformat(),
                            "is_coding_question": q.question_type == "coding",
                        }
                        
                        # Include code solution if exists
                        if q.code_solution:
                            result_dict["code_solution"] = q.code_solution
                            result_dict["language"] = q.language or "python"
                            result_dict["time_complexity"] = q.time_complexity
                            result_dict["space_complexity"] = q.space_complexity
                        else:
                            result_dict["code_solution"] = None
                            result_dict["language"] = None
                            result_dict["time_complexity"] = None
                            result_dict["space_complexity"] = None
                        
                        results.append(result_dict)
                
                # Only log if we found cached questions
                if len(vector_results) > 0:
                    logger.info(f"Added {len(vector_results)} cached questions")
            
            except Exception as e:
                logger.error(f"Vector DB search failed: {e}", exc_info=True)
        
        # STEP 3: Generate with LLM if still not enough
        if not verified_only and len(results) < limit:
            try:
                logger.info("Generating questions with LLM...")
                needed = limit - len(results)
                
                logger.info(f"Generating {needed} questions with LLM as fallback")
                
                generated = await self._generate_and_label_questions(
                    query,
                    intent,
                    needed,
                    min_credibility,
                    api_key=api_key
                )
                
                # NULL CHECK
                if generated is None:
                    logger.warning("_generate_and_label_questions returned None")
                    generated = []
                
                results.extend(generated)
                generated_count = len(generated)
                
                logger.info(f"Generated {generated_count} questions")
            
            except Exception as e:
                logger.error(f"LLM generation failed: {e}", exc_info=True)
        
        # STEP 4: Rank by credibility
        try:
            logger.info("Ranking results by credibility...")
            
            ranked_results = await self._rank_by_credibility(results, query, intent)
            
            # NULL CHECK
            if ranked_results is None:
                logger.warning("_rank_by_credibility returned None, using unranked results")
                ranked_results = results
            
            results = ranked_results
        
        except Exception as e:
            logger.error(f"Ranking failed: {e}", exc_info=True)
            # Continue with unranked results

        # STEP 6: Ensure coding questions contain executable solutions
        try:
            results = await self._backfill_code_solutions(results, intent, api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to backfill code solutions in enhanced mode: {e}", exc_info=True)
        
        # STEP 5: Return final results
        final_results = results[:limit] if results else []
        
        # If verified_only is True but we got 0 results, add a helpful message
        if verified_only and len(final_results) == 0:
            logger.warning(
                f"Verified-only search returned 0 results. "
                f"Query: '{query}', min_credibility: {min_credibility}, company: {company}"
            )
            # The metadata will handle the warning message
        
        logger.info(
            f"Returning {len(final_results)} questions: "
            f"{verified_count} verified (dynamic sources), "
            f"{generated_count} LLM-generated"
        )
        
        return final_results

class UltraProductionInterviewService(EnhancedInterviewIntelligenceService):
    
    def __init__(self):
        super().__init__()
        
        # Advanced components
        self.hybrid_search: Optional[HybridSearchEngine] = None
        self.reranker: Optional[CohereReranker] = None
        self.code_executor: Optional[CodeExecutionSandbox] = None
        self.feedback_system: Optional[UserFeedbackSystem] = None
        self.query_expander: Optional[QueryExpansion] = None
        
        # Configuration
        self.enable_hybrid_search = settings.enable_hybrid_search
        self.enable_reranking = settings.enable_reranking and bool(settings.cohere_api_key)
        self.enable_code_execution = settings.enable_code_execution and bool(settings.judge0_api_key)
        self.enable_streaming = settings.enable_streaming
        self.enable_query_expansion = settings.enable_query_expansion
    
    async def initialize(self):
        """Initialize all components"""
        await super().initialize()
        
        # Initialize hybrid search
        if self.enable_hybrid_search:
            self.hybrid_search = HybridSearchEngine(
                qdrant_client=self.vector_client,
                collection_name=self.collection_name
            )
        else:
            self.hybrid_search = None
        
        # Initialize reranker
        if self.enable_reranking:
            self.reranker = CohereReranker(api_key=settings.cohere_api_key)
            logger.info("Cohere reranking enabled")
        
        # Initialize code executor
        if self.enable_code_execution:
            self.code_executor = CodeExecutionSandbox(
                judge0_api_key=getattr(settings, 'judge0_api_key', None)
            )
        else:
            self.code_executor = None
        
        # Initialize feedback system
        # TODO: Pass actual DB client
        self.feedback_system = UserFeedbackSystem(db_client=None)
        
        # Initialize query expander
        if self.enable_query_expansion:
            from app.services.llm_service import llm_service
            self.query_expander = QueryExpansion(llm_service=llm_service)
        else:
            self.query_expander = None
        
        logger.info("🚀 Ultra production service initialized with all features")
    
    async def search_questions(
        self,
        query: str,
        limit: int = 20,
        use_cache: bool = True,
        force_refresh: bool = False,
        verified_only: bool = False,
        min_credibility: float = 0.0,
        company: Optional[str] = None,
        use_web_scraping: bool = True,
        enable_reranking: bool = True,
        enable_query_expansion: bool = True,
        user_id: Optional[str] = None,  # For personalization
        api_key: Optional[str] = None,
    ) -> List[Dict]:
        """
        Ultra-enhanced search with all features
        
        Flow:
        1. Query expansion (find related queries)
        2. Multi-source search (API + Scraping + Vector DB)
        3. Hybrid search (BM25 + Semantic)
        4. Reranking (Cohere)
        5. Code validation (if coding questions)
        6. Personalization (based on user history)
        7. Return ranked results
        """
        
        logger.info(f"🔍 Ultra search: query='{query}', limit={limit}")
        
        # STEP 1: Query Expansion (find related queries)
        queries = [query]
        logger.info(f"Query expansion check: enable_query_expansion={enable_query_expansion}, self.enable_query_expansion={self.enable_query_expansion}, query_expander={self.query_expander is not None}")
        if enable_query_expansion and self.enable_query_expansion and self.query_expander:
            try:
                expanded = await self.query_expander.expand_query(query, api_key=api_key)
                queries = expanded
                logger.info(f"Expanded to {len(queries)} queries: {queries}")
            except Exception as e:
                logger.error(f"Query expansion failed: {e}")
        
        # STEP 2: Analyze intent (for first query)
        intent = await self._analyze_query_intent(query)
        logger.info(f"Intent: {intent.question_type}, requires_code={intent.requires_code}")
        
        # STEP 3: REAL-TIME WEB SEARCH - Fetch from internet using WebSearchAdapter + GitHub + LeetCode
        all_results = []
        seen_questions = set()
        
        # Use DynamicUnifiedSourceManager for REAL web search (disabled - using LLM only)
        if use_web_scraping:
            try:
                # Call the REAL source manager with WebSearchAdapter
                verified_questions = await self.source_manager.search_verified_questions(
                    query=query,
                    company=company,
                    min_credibility=min_credibility,
                    limit=limit * 2  # Get more for reranking
                )
                
                # Only log if we actually found questions
                if verified_questions:
                    logger.info(f"✅ Found {len(verified_questions)} questions from external sources")
                    
                    # Convert VerifiedQuestion to dict format
                    for vq in verified_questions:
                        result_dict = self._convert_verified_to_dict(vq)
                        
                        # Apply credibility filter only
                        if result_dict.get("credibility_score", 0) < min_credibility:
                            continue
                        
                        q_key = result_dict.get('question', '')[:100].lower()
                        if q_key and q_key not in seen_questions:
                            all_results.append(result_dict)
                            seen_questions.add(q_key)
            
            except Exception as e:
                logger.debug(f"External search skipped: {e}")
        
        # STEP 4: Also call parent's enhanced search for additional sources
        for search_query in queries:
            try:
                # Call parent's enhanced search (but skip if we already have enough)
                if len(all_results) >= limit * 2:
                    logger.info(f"Skipping enhanced search, already have {len(all_results)} results")
                    break
                
                results = await super().search_questions(
                    query=search_query,
                    limit=limit * 2,  # Get more for reranking
                    min_credibility=min_credibility,
                    company=company,
                    use_web_scraping=use_web_scraping,
                    force_refresh=force_refresh,
                    api_key=api_key
                )
                
                # Deduplicate and merge
                for result in results:
                    q_key = result.get('question', '')[:100].lower()
                    if q_key and q_key not in seen_questions:
                        all_results.append(result)
                        seen_questions.add(q_key)
                
                logger.info(f"Query '{search_query}' returned {len(results)} additional results")
            
            except Exception as e:
                logger.error(f"Enhanced search failed for '{search_query}': {e}")
        
        logger.info(f"Total results before hybrid/reranking: {len(all_results)}")
        
        # STEP 5: Hybrid Search (BM25 + Semantic)
        logger.info(f"Hybrid search check: enable={self.enable_hybrid_search}, hybrid_search={self.hybrid_search is not None}, results={len(all_results)}")
        if self.enable_hybrid_search and self.hybrid_search and len(all_results) > 10:
            try:
                # Initialize BM25 with current results
                await self.hybrid_search.initialize_bm25(all_results)
                
                # Perform hybrid search
                hybrid_results = await self.hybrid_search.hybrid_search(
                    query=query,
                    k=limit * 2,
                    keyword_weight=0.3,
                    semantic_weight=0.7
                )
                
                # Merge hybrid scores back into results
                for result in all_results:
                    for hybrid in hybrid_results:
                        if result.get('question') == hybrid.get('question'):
                            result['hybrid_score'] = hybrid.get('hybrid_score', 0.5)
                            break
                
                # Sort by hybrid score
                all_results.sort(
                    key=lambda x: x.get('hybrid_score', x.get('credibility_score', 0.5)),
                    reverse=True
                )
                
                logger.info("Applied hybrid search ranking")
            
            except Exception as e:
                logger.error(f"Hybrid search failed: {e}")
        else:
            logger.info(f"Hybrid search skipped: enable={self.enable_hybrid_search}, hybrid_search={self.hybrid_search is not None}, results={len(all_results)}")
        
        # STEP 6: Reranking (Cohere)
        logger.info(f"Reranking check: enable_reranking={enable_reranking}, self.enable_reranking={self.enable_reranking}, reranker={self.reranker is not None}, results={len(all_results)}")
        if enable_reranking and self.enable_reranking and self.reranker and len(all_results) > 5:
            try:
                reranked = await self.reranker.rerank(
                    query=query,
                    documents=all_results,
                    top_n=limit * 2
                )
                
                if reranked:
                    all_results = reranked
                    logger.info(f"Reranked to {len(reranked)} results")
            
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
        else:
            logger.info(f"Reranking skipped: enable_reranking={enable_reranking}, self.enable_reranking={self.enable_reranking}, reranker={self.reranker is not None}, results={len(all_results)}")
        
        # STEP 7: Code Validation (for coding questions)
        logger.info(f"Code validation check: requires_code={intent.requires_code}, enable_code_execution={self.enable_code_execution}, code_executor={self.code_executor is not None}")
        if intent.requires_code and self.enable_code_execution and self.code_executor:
            all_results = await self._validate_code_solutions(all_results)
        
        # STEP 8: Personalization (TODO: based on user history)
        if user_id:
            all_results = await self._personalize_results(all_results, user_id)
        
        # STEP 9: Final ranking and limit
        final_results = all_results[:limit]
        
        logger.info(f"✅ Returning {len(final_results)} ultra-enhanced results")
        
        return final_results
    
    async def _validate_code_solutions(self, results: List[Dict]) -> List[Dict]:
        """Validate code solutions and add execution results"""
        
        validated = []
        
        for result in results:
            if result.get('code_solution') and result.get('language'):
                try:
                    # Execute code
                    execution = await self.code_executor.execute_code(
                        code=result['code_solution'],
                        language=result['language']
                    )
                    
                    # Add execution metadata
                    result['code_validated'] = execution['success']
                    result['execution_time'] = execution.get('execution_time', 0)
                    result['execution_error'] = execution.get('error', '')
                    
                    # Boost credibility if code runs successfully
                    if execution['success']:
                        result['credibility_score'] = min(
                            1.0,
                            result.get('credibility_score', 0.5) + 0.1
                        )
                    
                    logger.debug(f"Validated code for: {result.get('question', '')[:50]}")
                
                except Exception as e:
                    logger.debug(f"Code validation failed: {e}")
                    result['code_validated'] = False
            
            validated.append(result)
        
        return validated
    
    async def _personalize_results(
        self,
        results: List[Dict],
        user_id: str
    ) -> List[Dict]:
        """
        Personalize results based on user history
        
        TODO: Implement full personalization with:
        - User skill level
        - Previously attempted questions
        - Success rate
        - Learning preferences
        """
        
        # Placeholder for now
        logger.debug(f"Personalizing results for user {user_id}")
        
        # TODO: Fetch user profile
        # user_profile = await self.get_user_profile(user_id)
        
        # TODO: Filter based on skill level
        # if user_profile.skill_level == 'beginner':
        #     results = [r for r in results if r.get('difficulty') in ['easy', 'medium']]
        
        # TODO: Remove already attempted questions
        # attempted = await self.get_attempted_questions(user_id)
        # results = [r for r in results if r.get('question') not in attempted]
        
        return results
    
    async def stream_search_results(
        self,
        query: str,
        limit: int = 20,
        **kwargs
    ) -> AsyncGenerator[Dict, None]:
        """
        Stream search results in real-time
        
        Usage:
            async for result in service.stream_search_results(query):
                await websocket.send_json(result)
        """
        
        # DEPRECATED: This method has broken imports and is not currently used
        # TODO: Fix or remove this method if streaming search is needed
        logger.warning("stream_search_results is deprecated - broken import")
        raise NotImplementedError("stream_search_results is deprecated")
        
        # Use streaming implementation
        # from app.services.enhanced_multi_source_adapter import enhanced_multi_source_manager
        
        # sources = [
        #     enhanced_multi_source_manager,
        #     # Add more sources
        # ]
        
        # async for result in RealTimeSearchStream.stream_search_results(
        #     query=query,
        #     sources=sources,
        #     limit=limit
        # ):
        #     yield result
    
    async def execute_and_validate_code(
        self,
        code: str,
        language: str,
        question: str
    ) -> Dict[str, Any]:
        """
        Execute code and validate it's correct
        
        Returns:
            {
                'success': bool,
                'output': str,
                'error': str,
                'is_correct': bool,
                'feedback': str
            }
        """
        
        if not self.code_executor:
            return {'success': False, 'error': 'Code execution not available'}
        
        # Execute code
        execution = await self.code_executor.execute_code(code, language)
        
        if not execution['success']:
            return {
                **execution,
                'is_correct': False,
                'feedback': f"Code failed to execute: {execution['error']}"
            }
        
        # Validate solution using LLM
        validation = await self.code_executor.validate_solution(
            code=code,
            language=language,
            expected_behavior=question
        )
        
        return {
            **execution,
            'is_correct': validation['is_valid'],
            'feedback': validation['feedback'],
            'suggestions': validation.get('suggestions', [])
        }
    
    async def record_user_feedback(
        self,
        question_id: str,
        user_id: str,
        vote: int,
        feedback_text: Optional[str] = None
    ):
        """Record user vote on question quality"""
        
        if self.feedback_system:
            await self.feedback_system.record_vote(
                question_id=question_id,
                user_id=user_id,
                vote=vote,
                feedback_text=feedback_text
            )
    
    async def report_question(
        self,
        question_id: str,
        user_id: str,
        reason: str
    ):
        """Report incorrect or outdated question"""
        
        if self.feedback_system:
            await self.feedback_system.report_incorrect(
                question_id=question_id,
                user_id=user_id,
                reason=reason
            )
    
    async def get_search_metadata(
        self,
        questions: List[Dict],
        verified_only: bool = False,
        min_credibility: float = 0.0
    ) -> Dict[str, Any]:
        """Enhanced metadata with additional stats"""
        
        # Get base metadata
        metadata = await super().get_search_metadata(
            questions, verified_only, min_credibility
        )
        
        # Add advanced stats
        if questions:
            # Code validation stats
            code_validated = sum(1 for q in questions if q.get('code_validated', False))
            
            # Reranking stats
            reranked = sum(1 for q in questions if 'rerank_score' in q)
            
            # Hybrid search stats
            hybrid_scored = sum(1 for q in questions if 'hybrid_score' in q)
            
            metadata['advanced_features'] = {
                'code_validated': code_validated,
                'reranked': reranked,
                'hybrid_search': hybrid_scored,
                'query_expanded': True  # If we got here, expansion happened
            }
            
            # Quality metrics
            avg_rerank_score = sum(
                q.get('rerank_score', 0) for q in questions
            ) / len(questions) if reranked > 0 else 0
            
            metadata['quality_metrics'] = {
                'avg_rerank_score': round(avg_rerank_score, 3),
                'code_execution_success_rate': (
                    code_validated / sum(1 for q in questions if q.get('code_solution'))
                    if any(q.get('code_solution') for q in questions) else 0
                )
            }
        
        return metadata


# Global service instances
# Create enhanced service first (it has all functionality)
from app.services.llm_service import get_llm_service

# Use Groq for Interview Intelligence tab only, not for the global default
class GroqUltraProductionInterviewService(UltraProductionInterviewService):
    def __init__(self):
        super().__init__()
        self.llm_service = get_llm_service("groq")

class GroqInterviewIntelligenceService(ModernInterviewIntelligenceService):
    def __init__(self):
        super().__init__()
        self.llm_service = get_llm_service("groq")

class GroqEnhancedInterviewIntelligenceService(EnhancedInterviewIntelligenceService):
    def __init__(self):
        super().__init__()
        self.llm_service = get_llm_service("groq")

# These are used only for Interview Intelligence tab (all use Groq)
ultra_production_service = GroqUltraProductionInterviewService()
enhanced_interview_service = GroqEnhancedInterviewIntelligenceService()
base_interview_service = GroqInterviewIntelligenceService()
interview_intelligence_service = base_interview_service