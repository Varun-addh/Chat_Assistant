import asyncio
import aiohttp
import json
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Set
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
import logging

logger = logging.getLogger(__name__)

from app.utils.time import utcnow


# ============================================================================
# CORE DATA MODELS
# ============================================================================

class QuestionDomain(str, Enum):
    """Question domains for routing"""
    CODING = "coding"
    SYSTEM_DESIGN = "system_design"
    BEHAVIORAL = "behavioral"
    DEVOPS = "devops"
    DATA_ENGINEERING = "data_engineering"
    CLOUD = "cloud"
    SECURITY = "security"
    FRONTEND = "frontend"
    BACKEND = "backend"
    MOBILE = "mobile"
    ML_AI = "ml_ai"
    DATABASE = "database"
    NETWORKING = "networking"
    GENERAL_TECHNICAL = "general_technical"


class SourceType(str, Enum):
    """Source types with credibility"""
    GITHUB_CURATED = "github_curated"
    LLM_GENERATED = "llm_generated"


class VerificationStatus(str, Enum):
    VERIFIED_REAL = "verified_real"
    LIKELY_REAL = "likely_real"
    LLM_GENERATED = "llm_generated"


class VerifiedQuestion(BaseModel):
    """Universal question model"""
    question: str
    answer: str
    topic: str
    difficulty: str
    question_type: str
    domain: QuestionDomain
    
    source_type: SourceType
    verification_status: VerificationStatus
    source_url: Optional[str] = None
    source_platform: Optional[str] = None
    
    company: Optional[str] = None
    companies: List[str] = Field(default_factory=list)
    
    key_concepts: List[str] = Field(default_factory=list)
    common_mistakes: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    
    # For coding
    code_solution: Optional[str] = None
    language: Optional[str] = None
    time_complexity: Optional[str] = None
    space_complexity: Optional[str] = None
    
    # Metadata
    credibility_score: float = Field(default=0.5, ge=0.0, le=1.0)
    frequency_score: float = Field(default=1.0, ge=0.0, le=10.0)
    created_at: datetime = Field(default_factory=utcnow)
    
    model_config = ConfigDict(use_enum_values=True)


# ============================================================================
# DOMAIN DETECTION
# ============================================================================

class QueryRouter:
    """Detect question domain from query"""
    
    DOMAIN_KEYWORDS = {
        QuestionDomain.DATABASE: [
            'sql', 'database', 'nosql', 'postgresql', 'mysql', 'mongodb', 'redis'
        ],
        QuestionDomain.SYSTEM_DESIGN: [
            'system design', 'architecture', 'scalability', 'microservice', 'distributed'
        ],
        QuestionDomain.ML_AI: [
            'machine learning', 'ai', 'neural network', 'deep learning', 'model'
        ],
        QuestionDomain.DEVOPS: [
            'devops', 'docker', 'kubernetes', 'ci/cd', 'jenkins', 'terraform'
        ],
        QuestionDomain.CLOUD: [
            'aws', 'azure', 'gcp', 'cloud', 'lambda', 'ec2', 's3'
        ],
        QuestionDomain.DATA_ENGINEERING: [
            'spark', 'hadoop', 'etl', 'data pipeline', 'airflow', 'kafka'
        ],
        QuestionDomain.FRONTEND: [
            'react', 'vue', 'angular', 'javascript', 'css', 'html', 'frontend'
        ],
        QuestionDomain.BACKEND: [
            'backend', 'api', 'rest', 'graphql', 'node', 'django', 'flask'
        ],
        QuestionDomain.CODING: [
            'algorithm', 'data structure', 'leetcode', 'coding', 'array', 'tree', 'graph'
        ],
    }
    
    @staticmethod
    def detect_domain(query: str) -> QuestionDomain:
        """Detect the domain from query keywords"""
        query_lower = query.lower()
        
        # Score each domain
        scores: Dict[QuestionDomain, int] = {}
        for domain, keywords in QueryRouter.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[domain] = score
        
        # Return highest scoring domain
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return QuestionDomain.GENERAL_TECHNICAL


# ============================================================================
# GITHUB SEARCH (SIMPLIFIED)
# ============================================================================

class GitHubSearcher:
    """Simple GitHub searcher without complex adapter pattern"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, List[VerifiedQuestion]] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=24)
    
    async def _ensure_session(self):
        """Create session with GitHub auth if available"""
        if not self.session or self.session.closed:
            import os
            github_token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_API_KEY")
            
            headers = {}
            if github_token:
                headers["Authorization"] = f"token {github_token}"
                logger.info("Using GitHub authentication token")
            
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def search(
        self,
        query: str,
        domain: QuestionDomain,
        limit: int = 20
    ) -> List[VerifiedQuestion]:
        """Search GitHub for interview questions"""
        try:
            # Check cache
            cache_key = f"{query}_{domain.value}_{limit}"
            if cache_key in self._cache and self._cache_time:
                if utcnow() - self._cache_time < self._cache_ttl:
                    logger.info(f"✅ Cache hit for GitHub search")
                    return self._cache[cache_key][:limit]
            
            session = await self._ensure_session()
            
            # Build search query
            domain_term = domain.value.replace("_", " ")
            search_query = f"interview questions {query} {domain_term}"
            
            # GitHub API search
            search_url = "https://api.github.com/search/code"
            params = {
                "q": f"{search_query} extension:md in:file interview",
                "sort": "indexed",
                "order": "desc",
                "per_page": min(limit, 10)
            }
            
            logger.info(f"Searching GitHub: {params['q']}")
            
            async with session.get(search_url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    logger.warning(f"GitHub API returned status {resp.status}")
                    return []
                
                data = await resp.json()
                items = data.get("items", [])
                logger.info(f"GitHub found {data.get('total_count', 0)} files, got {len(items)} items")
                
                questions = []
                for item in items[:5]:  # Process top 5 files
                    file_questions = await self._extract_questions_from_file(item, domain, session)
                    questions.extend(file_questions)
                    if len(questions) >= limit:
                        break
                
                # Cache results
                self._cache[cache_key] = questions
                self._cache_time = utcnow()
                
                logger.info(f"GitHub search extracted {len(questions)} total questions")
                return questions[:limit]
                
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
            return []
    
    async def _extract_questions_from_file(
        self,
        item: dict,
        domain: QuestionDomain,
        session: aiohttp.ClientSession
    ) -> List[VerifiedQuestion]:
        """Extract questions from a GitHub file"""
        try:
            # Get raw file content
            raw_url = item.get("html_url", "").replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            
            if not raw_url:
                return []
            
            async with session.get(raw_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return []
                
                content = await resp.text()
                
                # Extract Q&A pairs using regex
                questions = []
                
                # Pattern 1: **Q:** or **Question:**
                qa_pattern = r'\*\*Q(?:uestion)?:\*\*\s*(.+?)(?:\*\*A(?:nswer)?:\*\*\s*(.+?))?(?=\*\*Q|\Z)'
                matches = re.findall(qa_pattern, content, re.DOTALL | re.IGNORECASE)
                
                for q_text, a_text in matches[:5]:
                    q_text = q_text.strip()
                    a_text = a_text.strip() if a_text else "See source for answer"
                    
                    if len(q_text) > 20 and len(q_text) < 500:
                        questions.append(VerifiedQuestion(
                            question=q_text[:300],
                            answer=a_text[:1000] if a_text else f"Source: {item.get('html_url', '')}",
                            topic=domain.value,
                            difficulty="medium",
                            question_type="technical",
                            domain=domain,
                            source_type=SourceType.GITHUB_CURATED,
                            verification_status=VerificationStatus.VERIFIED_REAL,
                            source_url=item.get("html_url"),
                            source_platform="GitHub",
                            credibility_score=0.85,
                            frequency_score=5.0
                        ))
                
                logger.info(f"Extracted {len(questions)} questions from {item.get('name', 'file')}")
                return questions
                
        except Exception as e:
            logger.debug(f"Failed to extract from file: {e}")
            return []
    
    async def close(self):
        """Close session"""
        if self.session and not self.session.closed:
            await self.session.close()


# ==========================================================================
# BACKWARDS-COMPAT: WebSearchAdapter
# ==========================================================================

class WebSearchAdapter:
    """Compatibility adapter used by older scripts/tests.

    The project evolved away from an adapter-per-source design, but a number of
    entrypoints still import `WebSearchAdapter`. This implementation keeps the
    public contract while delegating to the simplified GitHub searcher.
    """

    def __init__(self):
        self._github = GitHubSearcher()

    async def search(
        self,
        query: str,
        domain: QuestionDomain,
        company: Optional[str] = None,
        limit: int = 20,
    ) -> List[VerifiedQuestion]:
        # `company` is currently unused by the GitHub search implementation.
        _ = company
        return await self._github.search(query=query, domain=domain, limit=limit)

    async def close(self) -> None:
        await self._github.close()


# ============================================================================
# UNIFIED SOURCE MANAGER (SIMPLIFIED)
# ============================================================================

class DynamicUnifiedSourceManager:
    """
    Simplified source manager - No external sources, pure LLM generation
    All adapters (LeetCode/GitHub/DevOps/ML) removed
    """
    
    def __init__(self):
        self.router = QueryRouter()
    
    async def search_verified_questions(
        self,
        query: str,
        company: Optional[str] = None,
        min_credibility: float = 0.0,
        limit: int = 20
    ) -> List[VerifiedQuestion]:
        """
        Return empty list - no external sources, rely on LLM generation
        """
        
        # Detect domain (for logging purposes)
        domain = self.router.detect_domain(query)
        logger.debug(f"Domain: {domain}, external sources disabled")
        
        # Return empty list - let LLM generate all questions
        return []
    
    async def get_source_info(self) -> Dict[str, Any]:
        """Get information about available sources"""
        return {
            "total_sources": 0,
            "sources": {
                "LLM": "All questions generated by AI"
            }
        }
    
    async def close(self):
        """Close all resources"""
        pass  # No external resources to close


# ============================================================================
# GLOBAL INSTANCE (for backward compatibility)
# ============================================================================

# Create a global instance that other modules can import
dynamic_source_manager = DynamicUnifiedSourceManager()
