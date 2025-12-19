import asyncio
import aiohttp
import json
import re
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
import logging

from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Qdrant client for production
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# Pydantic models
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# 1. HYBRID SEARCH (Keyword + Semantic)
# ============================================================================

class HybridSearchEngine:
    """
    Combines BM25 (keyword search) + FAISS/Qdrant (semantic search)
    for better retrieval accuracy
    """
    
    def __init__(self, qdrant_client: QdrantClient, collection_name: str):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        import os
        MODEL_PATH = str((__file__ and __file__) and (os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/models/all-MiniLM-L6-v2'))))
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/models'))
        self.embeddings = HuggingFaceEmbeddings(model_name=MODEL_PATH)
        
        # BM25 for keyword search
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.documents: List[Document] = []
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    async def initialize_bm25(self, questions: List[Dict]):
        """Initialize BM25 retriever with questions"""
        self.documents = []
        
        for q in questions:
            content = f"""
            Question: {q.get('question', '')}
            Answer: {q.get('answer', '')}
            Topic: {q.get('topic', '')}
            Concepts: {', '.join(q.get('key_concepts', []))}
            Difficulty: {q.get('difficulty', '')}
            """
            
            doc = Document(
                page_content=content.strip(),
                metadata={
                    'question': q.get('question'),
                    'source': q.get('source'),
                    'credibility': q.get('credibility_score', 0.5),
                    'question_type': q.get('question_type'),
                    'topic': q.get('topic'),
                }
            )
            self.documents.append(doc)
        
        # Create BM25 retriever
        if self.documents:
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)
            self.bm25_retriever.k = 20  # Top 20 results
            logger.info(f"Initialized BM25 with {len(self.documents)} documents")
    
    async def hybrid_search(
        self,
        query: str,
        k: int = 20,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> List[Dict]:
        """
        Perform hybrid search combining keyword and semantic search
        
        Args:
            query: Search query
            k: Number of results
            keyword_weight: Weight for BM25 (0.0-1.0)
            semantic_weight: Weight for semantic search (0.0-1.0)
        """
        
        results = []
        
        # 1. BM25 Keyword Search
        keyword_results = {}
        if self.bm25_retriever:
            try:
                # Use new LangChain API (invoke returns a list of Documents)
                bm25_docs = await asyncio.to_thread(
                    self.bm25_retriever.invoke,
                    query
                )
                
                for idx, doc in enumerate(bm25_docs[:k]):
                    question = doc.metadata.get('question', '')
                    if question:
                        # Score based on rank (1.0 for first, decreasing)
                        score = 1.0 - (idx / len(bm25_docs))
                        keyword_results[question] = {
                            'score': score * keyword_weight,
                            'doc': doc
                        }
                
                logger.info(f"BM25 found {len(keyword_results)} results")
            except Exception as e:
                logger.error(f"BM25 search failed: {e}")
        
        # 2. Semantic Search (Qdrant)
        semantic_results = {}
        try:
            # Create query embedding
            query_vector = self.embeddings.embed_query(query)
            
            # Search Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k,
                score_threshold=0.5
            )
            
            for result in search_results:
                question = result.payload.get('question', '')
                if question:
                    semantic_results[question] = {
                        'score': result.score * semantic_weight,
                        'payload': result.payload
                    }
            
            logger.info(f"Semantic search found {len(semantic_results)} results")
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
        
        # 3. Combine scores (Reciprocal Rank Fusion)
        combined_scores = {}
        all_questions = set(keyword_results.keys()) | set(semantic_results.keys())
        
        for question in all_questions:
            keyword_score = keyword_results.get(question, {}).get('score', 0.0)
            semantic_score = semantic_results.get(question, {}).get('score', 0.0)
            
            # Combine with weights
            combined_scores[question] = keyword_score + semantic_score
        
        # Sort by combined score
        sorted_questions = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get full question data
        for question, score in sorted_questions[:k]:
            # Try to get from keyword results first, then semantic
            if question in keyword_results:
                doc = keyword_results[question]['doc']
                result = {
                    'question': question,
                    'source': doc.metadata.get('source'),
                    'credibility': doc.metadata.get('credibility'),
                    'hybrid_score': score,
                    'retrieval_method': 'hybrid'
                }
            elif question in semantic_results:
                payload = semantic_results[question]['payload']
                result = {**payload, 'hybrid_score': score, 'retrieval_method': 'hybrid'}
            else:
                continue
            
            results.append(result)
        
        logger.info(f"Hybrid search returning {len(results)} results")
        return results


# ============================================================================
# 2. RERANKING with Cohere
# ============================================================================

class CohereReranker:
    """
    Rerank search results using Cohere's rerank API
    Significantly improves relevance
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.cohere.ai/v1/rerank"
    
    async def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_n: int = 20,
        model: str = "rerank-english-v3.0"
    ) -> List[Dict]:
        """
        Rerank documents using Cohere API
        
        Returns documents sorted by relevance with rerank scores
        """
        
        if not self.api_key:
            logger.warning("No Cohere API key, skipping reranking")
            return documents[:top_n]
        
        if not documents:
            return []
        
        try:
            # Prepare documents for reranking
            doc_texts = []
            for doc in documents:
                text = f"Question: {doc.get('question', '')}\n"
                text += f"Answer: {doc.get('answer', '')}...\n"
                text += f"Concepts: {', '.join(doc.get('key_concepts', []))}"
                doc_texts.append(text[:1000])  # Limit to 1000 chars
            
            # Call Cohere API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "query": query,
                        "documents": doc_texts,
                        "top_n": top_n,
                        "return_documents": False
                    },
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Cohere rerank failed: {response.status} - {error_text}")
                        return documents[:top_n]
                    
                    data = await response.json()
                    results = data.get('results', [])
                    
                    # Reorder documents based on rerank scores
                    reranked = []
                    for result in results:
                        idx = result['index']
                        relevance_score = result['relevance_score']
                        
                        if idx < len(documents):
                            doc = documents[idx].copy()
                            doc['rerank_score'] = relevance_score
                            doc['original_rank'] = idx
                            reranked.append(doc)
                    
                    logger.info(f"Reranked {len(reranked)} documents")
                    return reranked
        
        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            return documents[:top_n]


# ============================================================================
# 3. CODE EXECUTION SANDBOX
# ============================================================================

class CodeExecutionSandbox:
    """
    Execute code safely using Judge0 API or Piston API
    Validates code solutions and generates test cases
    """
    
    def __init__(self, judge0_api_key: Optional[str] = None):
        self.judge0_api_key = judge0_api_key
        self.judge0_url = "https://judge0-ce.p.rapidapi.com"
        self.piston_url = "https://emkc.org/api/v2/piston"
        
        # Language ID mapping for Judge0
        self.language_ids = {
            'python': 71,
            'javascript': 63,
            'java': 62,
            'cpp': 54,
            'c': 50,
            'go': 60,
            'rust': 73,
            'typescript': 74,
        }
    
    async def execute_code(
        self,
        code: str,
        language: str,
        test_cases: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Execute code and return results
        
        Args:
            code: Source code to execute
            language: Programming language
            test_cases: List of {input, expected_output}
        
        Returns:
            {
                'success': bool,
                'output': str,
                'error': str,
                'execution_time': float,
                'memory_used': int,
                'test_results': List[Dict]
            }
        """
        
        # Try Judge0 first (more features), fallback to Piston
        if self.judge0_api_key:
            return await self._execute_judge0(code, language, test_cases)
        else:
            return await self._execute_piston(code, language, test_cases)
    
    async def _execute_judge0(
        self,
        code: str,
        language: str,
        test_cases: Optional[List[Dict]]
    ) -> Dict[str, Any]:
        """Execute using Judge0 API (paid but better)"""
        
        try:
            language_id = self.language_ids.get(language.lower(), 71)
            
            async with aiohttp.ClientSession() as session:
                # Submit code
                async with session.post(
                    f"{self.judge0_url}/submissions",
                    headers={
                        "X-RapidAPI-Key": self.judge0_api_key,
                        "X-RapidAPI-Host": "judge0-ce.p.rapidapi.com",
                        "Content-Type": "application/json"
                    },
                    json={
                        "source_code": code,
                        "language_id": language_id,
                        "stdin": "",
                        "cpu_time_limit": 2,
                        "memory_limit": 128000
                    },
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status != 201:
                        return {'success': False, 'error': f'Submission failed: {response.status}'}
                    
                    data = await response.json()
                    token = data.get('token')
                
                # Wait for result (poll)
                for _ in range(10):  # Max 10 attempts
                    await asyncio.sleep(0.5)
                    
                    async with session.get(
                        f"{self.judge0_url}/submissions/{token}",
                        headers={
                            "X-RapidAPI-Key": self.judge0_api_key,
                            "X-RapidAPI-Host": "judge0-ce.p.rapidapi.com"
                        },
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        result = await response.json()
                        
                        status = result.get('status', {}).get('description')
                        if status not in ['In Queue', 'Processing']:
                            # Execution complete
                            return {
                                'success': status == 'Accepted',
                                'output': result.get('stdout', ''),
                                'error': result.get('stderr', '') or result.get('compile_output', ''),
                                'execution_time': result.get('time', 0),
                                'memory_used': result.get('memory', 0),
                                'status': status
                            }
                
                return {'success': False, 'error': 'Execution timeout'}
        
        except Exception as e:
            logger.error(f"Judge0 execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_piston(
        self,
        code: str,
        language: str,
        test_cases: Optional[List[Dict]]
    ) -> Dict[str, Any]:
        """Execute using Piston API (free but basic)"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.piston_url}/execute",
                    json={
                        "language": language.lower(),
                        "version": "*",
                        "files": [{"content": code}],
                        "stdin": "",
                        "args": []
                    },
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        return {'success': False, 'error': f'Piston failed: {response.status}'}
                    
                    result = await response.json()
                    
                    return {
                        'success': not result.get('run', {}).get('stderr'),
                        'output': result.get('run', {}).get('stdout', ''),
                        'error': result.get('run', {}).get('stderr', ''),
                        'execution_time': 0,  # Piston doesn't provide this
                        'memory_used': 0
                    }
        
        except Exception as e:
            logger.error(f"Piston execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def validate_solution(
        self,
        code: str,
        language: str,
        expected_behavior: str
    ) -> Dict[str, Any]:
        """
        Validate if code solution is correct using LLM
        
        Returns:
            {
                'is_valid': bool,
                'feedback': str,
                'suggestions': List[str]
            }
        """
        
        # Execute code first
        execution_result = await self.execute_code(code, language)
        
        if not execution_result['success']:
            return {
                'is_valid': False,
                'feedback': f"Code has errors: {execution_result['error']}",
                'suggestions': ['Fix syntax errors', 'Check variable names']
            }
        
        # TODO: Use LLM to validate logic
        # For now, just check if it runs
        return {
            'is_valid': True,
            'feedback': "Code executes successfully",
            'suggestions': []
        }


# ============================================================================
# 4. REAL-TIME UPDATES with WebSockets
# ============================================================================

class RealTimeSearchStream:
    """
    Stream search results in real-time as they're found
    Better UX - users see results immediately
    """
    
    @staticmethod
    async def stream_search_results(
        query: str,
        sources: List[Any],
        limit: int = 20
    ) -> AsyncGenerator[Dict, None]:
        """
        Stream results as they come in from different sources
        
        Usage:
            async for result in stream_search_results(query, sources):
                await websocket.send_json(result)
        """
        
        # Track what we've sent
        sent_questions = set()
        
        # Send initial message
        yield {
            'type': 'search_started',
            'query': query,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Search sources in parallel but yield as they complete
        tasks = []
        for source in sources:
            task = asyncio.create_task(source.search(query, limit=limit))
            tasks.append((source.__class__.__name__, task))
        
        # Yield results as they complete
        for source_name, task in tasks:
            try:
                # Send source status
                yield {
                    'type': 'source_searching',
                    'source': source_name,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                results = await task
                
                # Send each result
                for result in results:
                    question_key = result.get('question', '')[:100].lower()
                    
                    if question_key not in sent_questions:
                        sent_questions.add(question_key)
                        
                        yield {
                            'type': 'result',
                            'source': source_name,
                            'data': result,
                            'timestamp': datetime.utcnow().isoformat()
                        }
                
                # Send source complete
                yield {
                    'type': 'source_complete',
                    'source': source_name,
                    'count': len(results),
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            except Exception as e:
                yield {
                    'type': 'source_error',
                    'source': source_name,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
        
        # Send completion
        yield {
            'type': 'search_complete',
            'total_results': len(sent_questions),
            'timestamp': datetime.utcnow().isoformat()
        }


# ============================================================================
# 5. USER FEEDBACK SYSTEM
# ============================================================================

class UserFeedbackSystem:
    """
    Track user feedback to improve question quality over time
    Implements RLHF (Reinforcement Learning from Human Feedback)
    """
    
    def __init__(self, db_client):
        self.db = db_client
    
    async def record_vote(
        self,
        question_id: str,
        user_id: str,
        vote: int,  # +1 or -1
        feedback_text: Optional[str] = None
    ):
        """Record user vote on question quality"""
        
        vote_data = {
            'question_id': question_id,
            'user_id': user_id,
            'vote': vote,
            'feedback': feedback_text,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store in database
        # TODO: Implement actual DB storage
        logger.info(f"Recorded vote: {vote_data}")
        
        # Update question credibility score
        await self._update_credibility_score(question_id, vote)
    
    async def _update_credibility_score(self, question_id: str, vote: int):
        """Update question credibility based on votes"""
        
        # Get current votes
        total_votes = await self._get_vote_count(question_id)
        upvotes = await self._get_upvote_count(question_id)
        
        if total_votes > 0:
            # Calculate new credibility (0.0 - 1.0)
            vote_ratio = upvotes / total_votes
            
            # Blend with original credibility
            # More votes = more weight to user feedback
            vote_weight = min(total_votes / 100, 0.7)  # Max 70% weight
            original_weight = 1.0 - vote_weight
            
            # TODO: Update in vector DB
            new_credibility = (vote_ratio * vote_weight) + (0.5 * original_weight)
            
            logger.info(f"Updated credibility for {question_id}: {new_credibility:.2f}")
    
    async def _get_vote_count(self, question_id: str) -> int:
        # TODO: Implement actual DB query
        return 10
    
    async def _get_upvote_count(self, question_id: str) -> int:
        # TODO: Implement actual DB query
        return 7
    
    async def report_incorrect(
        self,
        question_id: str,
        user_id: str,
        reason: str
    ):
        """Report incorrect or outdated question"""
        
        report = {
            'question_id': question_id,
            'user_id': user_id,
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store report
        # TODO: Implement actual DB storage
        logger.warning(f"Question reported: {report}")
        
        # If multiple reports, mark for review
        report_count = await self._get_report_count(question_id)
        if report_count >= 3:
            await self._mark_for_review(question_id)
    
    async def _get_report_count(self, question_id: str) -> int:
        # TODO: Implement actual DB query
        return 1
    
    async def _mark_for_review(self, question_id: str):
        """Mark question for manual review"""
        logger.warning(f"Question {question_id} marked for review (multiple reports)")


# ============================================================================
# 6. QUERY EXPANSION for Better Results
# ============================================================================

class QueryExpansion:
    """
    Expand user queries to find more relevant results
    Uses LLM to generate related queries
    """
    
    def __init__(self, llm_service):
        self.llm = llm_service
    
    async def expand_query(self, query: str, api_key: Optional[str] = None) -> List[str]:
        """
        Generate related queries to improve coverage
        
        Example:
            "python coding questions" →
            [
                "python coding questions",
                "python interview problems",
                "python algorithm challenges",
                "python data structures questions"
            ]
        """
        
        if not self.llm.enabled:
            return [query]
        
        try:
            prompt = f"""Given this interview question search query: "{query}"

Generate 3 related search queries that would find similar questions.

CRITICAL: Return ONLY a valid JSON array of strings. No markdown, no explanation, no code fences.

Example format (copy this structure exactly):
["related query 1", "related query 2", "related query 3"]

Requirements:
- Return exactly 3 strings in a JSON array
- Use double quotes for all strings
- No trailing commas
- No markdown formatting"""
            
            response = await self.llm.generate_answer(prompt, api_key=api_key)
            
            if not response or not response.strip():
                return [query]
            
            # Clean response
            cleaned = response.strip()
            
            # Remove markdown code fences
            if "```json" in cleaned:
                try:
                    cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
                except Exception:
                    pass
            elif "```" in cleaned:
                try:
                    cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
                except Exception:
                    pass
            
            # Try to find JSON array
            array_match = re.search(r'\[.*?\]', cleaned, re.DOTALL)
            if array_match:
                cleaned = array_match.group(0)
            
            # Remove trailing commas
            cleaned = re.sub(r',(\s*[\]\}])', r'\1', cleaned)
            
            # Parse response
            try:
                expanded = json.loads(cleaned)
            except json.JSONDecodeError:
                # Try to extract strings manually
                string_matches = re.findall(r'"([^"]+)"', cleaned)
                if string_matches:
                    expanded = string_matches[:3]
                else:
                    # Last resort: split by common delimiters
                    parts = re.split(r'[,\n]', cleaned)
                    expanded = [p.strip().strip('"\'[]') for p in parts if p.strip()][:3]
            
            if isinstance(expanded, list) and len(expanded) > 0:
                # Filter out empty strings and ensure we have valid queries
                valid_expansions = [q for q in expanded[:3] if isinstance(q, str) and q.strip() and q.strip() != query]
                if valid_expansions:
                    return [query] + valid_expansions
            
            return [query]
        
        except Exception as e:
            logger.error(f"Query expansion failed: {e}", exc_info=True)
            return [query]


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'HybridSearchEngine',
    'CohereReranker',
    'CodeExecutionSandbox',
    'RealTimeSearchStream',
    'UserFeedbackSystem',
    'QueryExpansion'
]