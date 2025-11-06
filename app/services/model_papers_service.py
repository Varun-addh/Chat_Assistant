from __future__ import annotations

import json
import os
import re
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import hashlib

from app.schemas import ModelQuestion
from app.services.web_scraper_service import web_scraper_service
from app.services.llm_service import llm_service

# Optional embeddings support for fast semantic search
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None
    np = None


def _slugify_topic(topic: str) -> str:
    s = (topic or "").strip().lower()
    allowed = [c if c.isalnum() else "-" for c in s]
    slug = "".join(allowed).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "general"


class ModelPapersService:
    def __init__(self) -> None:
        self._sessions_dir = os.path.join("data", "sessions")
        self._cache_dir = os.path.join("data", "model_papers")
        self._staging_dir = os.path.join("data", "staging")
        self._embeddings_cache_dir = os.path.join("data", "embeddings_cache")
        os.makedirs(self._cache_dir, exist_ok=True)
        os.makedirs(self._staging_dir, exist_ok=True)
        os.makedirs(self._embeddings_cache_dir, exist_ok=True)
        
        # Lazy-load embeddings model (lightweight, fast)
        self._embedding_model: Optional[Any] = None
        self._embedding_model_loaded = False
        
        # In-memory cache for embeddings (key: text hash -> embedding vector)
        self._embeddings_cache: Dict[str, Any] = {}
        
        # Batch processing settings
        self._max_parallel_ai_calls = 10
        self._semantic_search_threshold = 0.65  # Cosine similarity threshold

        # Vector DB (Chroma) optional
        self._chroma = None
        self._chroma_collection = None

    def _get_embedding_model(self):
        """Lazy-load the embedding model only when needed."""
        if not _EMBEDDINGS_AVAILABLE:
            return None
        
        if not self._embedding_model_loaded:
            try:
                # Use lightweight, fast model optimized for semantic search
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self._embedding_model_loaded = True
            except Exception:
                self._embedding_model = None
        return self._embedding_model

    def _compute_embedding(self, text: str) -> Optional[Any]:
        """Compute embedding with caching."""
        if not text or not self._get_embedding_model():
            return None
        
        # Cache key based on text hash
        text_hash = hashlib.md5(text.lower().strip().encode()).hexdigest()
        
        # Check in-memory cache first
        if text_hash in self._embeddings_cache:
            return self._embeddings_cache[text_hash]
        
        # Check disk cache
        cache_file = os.path.join(self._embeddings_cache_dir, f"{text_hash}.npy")
        if os.path.exists(cache_file):
            embedding = np.load(cache_file)
            self._embeddings_cache[text_hash] = embedding
            return embedding
        
        # Compute embedding
        try:
            embedding = self._embedding_model.encode(text, convert_to_numpy=True)
            # Cache in memory and disk
            self._embeddings_cache[text_hash] = embedding
            np.save(cache_file, embedding)
            return embedding
        except Exception:
            return None

    def _cosine_similarity(self, vec1: Any, vec2: Any) -> float:
        """Compute cosine similarity between two vectors."""
        if vec1 is None or vec2 is None or np is None:
            return 0.0
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0

    def _read_json_file(self, path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_json_file(self, path: str, content: Dict[str, Any]) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)

    # ========== Vector DB (Chroma) ==========
    def _ensure_vectordb(self) -> None:
        if self._chroma is not None:
            return
        try:
            import chromadb
            self._chroma = chromadb.PersistentClient(path=os.path.join("data", "vector_db"))
        except Exception:
            self._chroma = None

    def _get_collection(self, topic: str):
        self._ensure_vectordb()
        if self._chroma is None:
            return None
        try:
            name = _slugify_topic(topic)
            self._chroma_collection = self._chroma.get_or_create_collection(name=f"model_papers_{name}")
        except Exception:
            self._chroma_collection = None
        return self._chroma_collection

    def _upsert_vectordb(self, topic: str, items: List[ModelQuestion]) -> None:
        col = self._get_collection(topic)
        if col is None:
            return
        try:
            ids = []
            docs = []
            metas = []
            for it in items[:200]:
                key = hashlib.md5((it.question or "").lower().encode()).hexdigest()
                ids.append(key)
                docs.append(f"Q: {it.question}\nA: {it.answer}")
                metas.append({"source": it.source or "", "updated_at": it.updated_at.isoformat()})
            if ids:
                col.upsert(ids=ids, documents=docs, metadatas=metas)
        except Exception:
            pass

    def _load_cached(self, topic: str) -> List[ModelQuestion]:
        slug = _slugify_topic(topic)
        path = os.path.join(self._cache_dir, f"{slug}.json")
        data = self._read_json_file(path)
        items: List[ModelQuestion] = []
        if not data:
            return items
        for it in data.get("items", []):
            try:
                items.append(
                    ModelQuestion(
                        question=it.get("question", ""),
                        answer=it.get("answer", ""),
                        source=it.get("source"),
                        updated_at=datetime.fromisoformat(it.get("updated_at")),
                    )
                )
            except Exception:
                continue
        return items

    def _save_cache(self, topic: str, items: List[ModelQuestion]) -> None:
        slug = _slugify_topic(topic)
        path = os.path.join(self._cache_dir, f"{slug}.json")
        payload = {
            "topic": topic,
            "updated_at": datetime.utcnow().isoformat(),
            "items": [
                {
                    "question": i.question,
                    "answer": i.answer,
                    "source": i.source,
                    "updated_at": i.updated_at.isoformat(),
                }
                for i in items
            ],
        }
        self._write_json_file(path, payload)

    def _keyword_match(self, text: str, topic: str) -> bool:
        """Fast keyword-based pre-filtering."""
        if not text or not topic:
            return False
        t = text.lower()
        topic_words = set(topic.lower().split())
        text_words = set(t.split())
        # Match if any topic word appears in text
        return len(topic_words & text_words) > 0

    async def _batch_ai_semantic_match(self, items: List[Dict[str, Any]], topic: str) -> List[Dict[str, Any]]:
        """Use AI to batch-evaluate relevance for multiple items in parallel."""
        if not llm_service.enabled or not items:
            return items
        
        # Use embeddings for fast semantic search if available
        if self._get_embedding_model():
            topic_embedding = self._compute_embedding(topic)
            if topic_embedding is not None:
                relevant_items = []
                for item in items:
                    # Combine question and answer for semantic matching
                    combined_text = f"{item['question']} {item['answer'][:500]}"
                    text_embedding = self._compute_embedding(combined_text)
                    if text_embedding is not None:
                        similarity = self._cosine_similarity(topic_embedding, text_embedding)
                        if similarity >= self._semantic_search_threshold:
                            item['_similarity'] = similarity
                            relevant_items.append(item)
                # Sort by similarity (highest first)
                relevant_items.sort(key=lambda x: x.get('_similarity', 0), reverse=True)
                return relevant_items
        
        # Fallback: Use LLM for batch relevance checking
        # Create a single prompt for multiple items
        items_text = "\n\n".join([
            f"Item {i+1}:\nQ: {item['question'][:300]}\nA: {item['answer'][:500]}"
            for i, item in enumerate(items[:20])  # Limit batch size
        ])
        
        prompt = (
            f"Analyze which of these interview Q&A pairs are relevant to the topic: '{topic}'\n\n"
            f"{items_text}\n\n"
            f"Respond with ONLY a JSON array of item numbers (1-indexed) that are relevant. "
            f"Example: [1, 3, 5]\n"
            f"Be lenient - include items that are tangentially related. Your response (JSON array only):"
        )
        
        try:
            response = await llm_service.generate_answer(
                prompt,
                system_prompt="You are a relevance classifier. Respond with a JSON array of relevant item numbers only.",
            )
            
            # Extract JSON array
            json_match = re.search(r'\[[\d\s,]+\]', response)
            if json_match:
                relevant_indices = set(json.loads(json_match.group()))
                return [
                    items[i-1] for i in relevant_indices 
                    if 1 <= i <= len(items)
                ]
        except Exception:
            pass
        
        # Fallback to keyword matching if AI fails
        return [
            item for item in items 
            if self._keyword_match(item['question'], topic) or self._keyword_match(item['answer'], topic)
        ]

    async def _collect_from_sessions(self, topic: str, horizon_days: int) -> List[ModelQuestion]:
        """Collect relevant Q&A from sessions using fast semantic search with embeddings."""
        items: List[ModelQuestion] = []
        now = datetime.utcnow()
        cutoff = now - timedelta(days=max(horizon_days, 0))

        if not os.path.isdir(self._sessions_dir):
            return items

        # Step 1: Fast collection of all candidate items (parallel file I/O)
        candidate_items: List[Dict[str, Any]] = []
        
        def _read_session_file(name: str) -> Optional[List[Dict[str, Any]]]:
            if not name.endswith(".json"):
                return None
            path = os.path.join(self._sessions_dir, name)
            data = self._read_json_file(path)
            if not data:
                return None
            
            session_id = data.get("session_id")
            session_items = []
            for qna in (data.get("qna") or []):
                q = (qna.get("question") or "").strip()
                a = (qna.get("answer") or "").strip()
                ts_raw = qna.get("created_at") or data.get("last_update")
                try:
                    ts = datetime.fromisoformat(ts_raw) if ts_raw else now
                except Exception:
                    ts = now
                if ts >= cutoff:
                    session_items.append({
                        "question": q,
                        "answer": a,
                        "source": session_id,
                        "updated_at": ts,
                    })
            return session_items
        
        # Parallel file reading
        import anyio
        file_names = [f for f in os.listdir(self._sessions_dir) if f.endswith(".json")]
        
        async def _read_all():
            tasks = [
                anyio.to_thread.run_sync(_read_session_file, name)
                for name in file_names[:100]  # Limit for performance
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        
        results = await _read_all()
        for result in results:
            if isinstance(result, list):
                candidate_items.extend(result)

        # Step 2: Fast keyword pre-filtering (reduces AI calls)
        keyword_filtered = [
            item for item in candidate_items
            if self._keyword_match(item['question'], topic) or self._keyword_match(item['answer'], topic)
        ]
        
        # Step 3: Semantic filtering using embeddings (fast) or batch AI (parallel)
        if keyword_filtered:
            # Process in parallel batches
            batch_size = 50
            batches = [
                keyword_filtered[i:i + batch_size]
                for i in range(0, len(keyword_filtered), batch_size)
            ]
            
            batch_tasks = [
                self._batch_ai_semantic_match(batch, topic)
                for batch in batches
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for batch_result in batch_results:
                if isinstance(batch_result, list):
                    for item in batch_result:
                        items.append(
                            ModelQuestion(
                                question=item['question'],
                                answer=item['answer'],
                                source=item['source'],
                                updated_at=item['updated_at'],
                                confidence=min(1.0, max(0.0, item.get('_similarity', 0.7))),
                            )
                        )

        # Sort by recency
        items.sort(key=lambda i: i.updated_at, reverse=True)
        return items

    async def _generate_ai_model_paper(self, topic: str, limit: int, context_items: Optional[List[ModelQuestion]] = None) -> List[ModelQuestion]:
        """Use AI to generate comprehensive model paper with structured output."""
        if not llm_service.enabled:
            return []

        # Build context from existing items if available
        context_text = ""
        if context_items:
            context_text = "\n\nExamples of existing questions:\n"
            for item in context_items[:5]:  # Use top 5 as examples
                context_text += f"Q: {item.question[:200]}\nA: {item.answer[:300]}\n\n"

        # Enhanced prompt with structured output instructions
        prompt = (
            f"Generate exactly {limit} diverse, real-world interview questions and comprehensive answers "
            f"for the topic: '{topic}'.\n\n"
            f"{context_text}"
            f"Requirements:\n"
            f"1. Include questions at varying difficulty levels (30% basic, 50% intermediate, 20% advanced)\n"
            f"2. Cover multiple aspects: concepts, implementation details, best practices, trade-offs, real-world scenarios\n"
            f"3. Answers must be comprehensive, interview-ready, and detailed (similar in quality to the examples above)\n"
            f"4. Ensure diversity - avoid similar questions\n"
            f"5. Format as a STRICT JSON array with this EXACT structure:\n"
            f"   {{\n"
            f"     \"questions\": [\n"
            f"       {{\"question\": \"...\", \"answer\": \"...\"}},\n"
            f"       {{\"question\": \"...\", \"answer\": \"...\"}}\n"
            f"     ]\n"
            f"   }}\n\n"
            f"CRITICAL: Output ONLY valid JSON. No markdown, no code blocks, no explanations. "
            f"Start with {{ and end with }}. The 'questions' array must contain exactly {limit} items."
        )

        try:
            response = await llm_service.generate_answer(
                prompt,
                system_prompt=(
                    "You are an expert interview question generator specializing in creating comprehensive, "
                    "interview-ready Q&A pairs. Always respond with valid JSON only, no markdown formatting."
                ),
            )
            
            # Extract JSON (handle various formats)
            json_text = response.strip()
            
            # Remove markdown code blocks if present
            if "```json" in json_text:
                json_text = re.search(r"```json\s*(.*?)\s*```", json_text, re.DOTALL)
                json_text = json_text.group(1) if json_text else response
            elif "```" in json_text:
                json_text = re.search(r"```\s*(.*?)\s*```", json_text, re.DOTALL)
                json_text = json_text.group(1) if json_text else response
            
            # Try to find JSON object
            json_match = re.search(r'\{[^{}]*"questions"\s*:\s*\[.*?\]\s*\}', json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            
            # Parse JSON
            data = json.loads(json_text)
            questions = data.get("questions", [])
            if not questions and isinstance(data, list):
                questions = data
            
            now = datetime.utcnow()
            items = []
            for item in questions[:limit]:
                q = (item.get("question") or "").strip()
                a = (item.get("answer") or "").strip()
                if q and a:
                    items.append(
                        ModelQuestion(
                            question=q,
                            answer=a,
                            source="ai_generated",
                            updated_at=now,
                            confidence=0.6,
                        )
                    )
            return items
        except Exception:
            # On error, return empty list (graceful degradation)
            return []

    async def _answer_scraped_questions(self, questions: List[Dict[str, Any]], topic: str) -> List[ModelQuestion]:
        """Batch-answer scraped questions via LLM with strict JSON output, preserving URLs/PDFs."""
        if not questions or not llm_service.enabled:
            return []
        # Limit batch size for stability
        questions = questions[:30]

        # Build prompt for batch answering
        q_texts = [ (item.get("question") or "").strip() for item in questions ]
        q_list = "\n".join([f"- {q}" for q in q_texts])
        prompt = (
            f"Provide comprehensive, interview-ready answers for the following questions on '{topic}'.\n\n"
            f"Questions:\n{q_list}\n\n"
            f"Output STRICT JSON with this structure:\n"
            f"{{\n  \"answers\": [\n    {{\"question\": \"...\", \"answer\": \"...\"}},\n    ...\n  ]\n}}\n\n"
            f"Output only valid JSON. No markdown."
        )

        try:
            response = await llm_service.generate_answer(
                prompt,
                system_prompt=(
                    "You are an expert interviewer. Generate comprehensive, accurate answers. "
                    "Always output valid JSON with the requested schema."
                ),
            )
            text = response.strip()
            if "```json" in text:
                m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
                text = m.group(1) if m else text
            elif "```" in text:
                m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
                text = m.group(1) if m else text

            mobj = re.search(r'\{[^{}]*"answers"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
            if mobj:
                text = mobj.group(0)
            data = json.loads(text)
            arr = data.get("answers", []) if isinstance(data, dict) else data
            now = datetime.utcnow()
            out: List[ModelQuestion] = []
            meta_by_q = { (item.get("question") or "").strip().lower(): item for item in questions }
            for item in arr:
                q = (item.get("question") or "").strip()
                a = (item.get("answer") or "").strip()
                if q and a:
                    meta = meta_by_q.get(q.lower(), {})
                    out.append(ModelQuestion(
                        question=q,
                        answer=a,
                        source="web_scraped+ai",
                        updated_at=now,
                        url=meta.get("url"),
                        pdf_url=meta.get("pdf_url"),
                    ))
            return out
        except Exception:
            return []

    async def get_model_paper(self, topic: str, limit: int = 20, include_recent_days: int = 30, use_ai_generation: bool = True) -> List[ModelQuestion]:
        """
        World-class model paper retrieval using:
        1. Fast embedding-based semantic search (if available)
        2. Parallel batch processing for AI calls
        3. Intelligent caching
        4. Structured AI generation
        """
        # Step 1: Load cached items (instant, no API calls)
        cached = self._load_cached(topic)
        
        # Step 2: Scrape the web for fresh questions, classify, and answer (scrape-first)
        scraped = web_scraper_service.scrape_topic(topic)
        # Stage raw scrape for observability/debugging
        try:
            stage_path = os.path.join(self._staging_dir, f"{_slugify_topic(topic)}_{int(datetime.utcnow().timestamp())}.json")
            self._write_json_file(stage_path, {"topic": topic, "items": [s.__dict__ for s in scraped]})
        except Exception:
            pass
        scraped_questions = [s.question for s in scraped]
        # Deduplicate quickly against cached questions (recent not loaded yet)
        existing_q_set_from_cache = { (i.question or '').strip().lower() for i in cached }
        new_scraped_questions = [q for q in scraped_questions if (q or '').strip().lower() not in existing_q_set_from_cache]

        # Use embeddings/AI to filter relevance (reusing batch matcher)
        # Preserve source URLs and PDF links for downstream
        meta_map = { (s.question or '').strip().lower(): {"url": s.url, "pdf_url": getattr(s, 'pdf_url', None)} for s in scraped }
        scraped_items_struct = []
        now_ts = datetime.utcnow()
        for q in new_scraped_questions:
            lower = (q or '').strip().lower()
            m = meta_map.get(lower, {})
            scraped_items_struct.append({
                "question": q,
                "answer": "",
                "source": "web",
                "updated_at": now_ts,
                "url": m.get("url"),
                "pdf_url": m.get("pdf_url"),
            })
        filtered_scraped = await self._batch_ai_semantic_match(scraped_items_struct, topic)
        # Keep full dicts to retain URLs/PDFs
        filtered_questions = filtered_scraped

        # Batch-answer scraped questions
        answered_scraped: List[ModelQuestion] = []
        if filtered_questions:
            answered_scraped = await self._answer_scraped_questions(filtered_questions, topic)
            # assign baseline confidence for reputable sources
            for it in answered_scraped:
                it.confidence = max(it.confidence or 0.0, 0.85)

        # Step 3: Collect recent items in parallel (fast semantic search) AFTER scraping
        recent = await self._collect_from_sessions(topic, include_recent_days)

        # Step 3: Deduplicate and merge
        seen: set[str] = set()
        merged: List[ModelQuestion] = []
        
        # Prioritize recent items (real interview questions)
        for item in recent:
            key = (item.question or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(item)
        
        # Add answered scraped items next
        for item in answered_scraped:
            key = (item.question or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(item)

        # Add cached items
        for item in cached:
            key = (item.question or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(item)

        # Step 4: Generate additional Q&A if needed (parallel processing)
        if use_ai_generation and llm_service.enabled and len(merged) < limit:
            needed = limit - len(merged)
            # Use existing items as context for better generation
            ai_generated = await self._generate_ai_model_paper(
                topic, 
                needed + 5,  # Generate extra to account for parsing
                context_items=merged[:5] if merged else None
            )
            
            # Add AI-generated items, avoiding duplicates
            for item in ai_generated:
                key = (item.question or "").strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    merged.append(item)

        # Step 5: Sort by priority (confidence desc, real>ai, newest first)
        def _rank_key(x: ModelQuestion):
            is_ai = 1 if (x.source or '').startswith('ai') else 0
            conf = x.confidence if x.confidence is not None else 0.5
            ts = x.updated_at.timestamp() if x.updated_at else 0
            return (-conf, is_ai, -ts)
        merged.sort(key=_rank_key)

        # Step 6: Trim to limit, upsert to vector DB, and cache
        trimmed = merged[: max(1, min(limit, 200))]
        try:
            self._upsert_vectordb(topic, trimmed)
        except Exception:
            pass
        # Cache top 200 items for faster future requests
        self._save_cache(topic, merged[:200])

        return trimmed


model_papers_service = ModelPapersService()
