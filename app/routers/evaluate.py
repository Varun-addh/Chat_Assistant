from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, Header
from fastapi.responses import JSONResponse
from datetime import datetime
import json
import hashlib
from typing import Dict, Optional
import logging

from app.utils.time import utcnow

from app.schemas import EvaluationIn, EvaluationOut, EvaluationScores, StaticSignals
from app.services.core.session_manager import get_session_manager
from app.services.core.code_evaluation_service import evaluate_code
from app.utils.audit import auditor
from app.services.chat.llm_service import get_llm_service
from app.middleware.auth import get_user_id_from_request
from app.services.core.redis_client import get_redis
from app.config import settings

from app.database import get_db_context
from app.utils.usage_tracking import track_api_usage

logger = logging.getLogger(__name__)

# Use intelligent provider selection with "copilot" feature flag for evaluation
llm_service = get_llm_service(feature="copilot")

router = APIRouter()

_EVAL_CACHE_PREFIX = "stratax:eval_cache:"
_EVAL_CACHE_TTL = getattr(settings, "redis_cache_ttl_seconds", 3600)

# In-memory fallback cache (used when Redis is unavailable)
_evaluation_cache: Dict[str, EvaluationOut] = {}


async def _cache_get(key: str) -> Optional[EvaluationOut]:
	"""Read evaluation result from Redis, falling back to in-memory cache."""
	try:
		redis = await get_redis()
		if redis is not None:
			raw = await redis.get(_EVAL_CACHE_PREFIX + key)
			if raw:
				return EvaluationOut.model_validate_json(raw)
			return None
	except Exception as e:
		logger.debug("Redis eval cache read error (using in-memory fallback): %s", e)
	return _evaluation_cache.get(key)


async def _cache_set(key: str, value: EvaluationOut) -> None:
	"""Write evaluation result to Redis (with TTL) and in-memory fallback."""
	try:
		redis = await get_redis()
		if redis is not None:
			await redis.setex(_EVAL_CACHE_PREFIX + key, _EVAL_CACHE_TTL, value.model_dump_json())
			return
	except Exception as e:
		logger.debug("Redis eval cache write error (using in-memory fallback): %s", e)
	_evaluation_cache[key] = value


async def _classify_evaluation_allowed(session_state, problem: Optional[str], code: Optional[str] = None, api_key: Optional[str] = None) -> tuple[bool, str]:
	"""Return (allowed, reason) whether evaluation should be permitted.
	Uses AI classification to avoid brittle keyword hardcoding.
	"""
	recent_qna = session_state.qna[-2:] if session_state.qna else []
	last_q_text = ""
	last_a_text = ""
	if recent_qna:
		last_item = recent_qna[-1]
		if isinstance(last_item, dict):
			last_q_text = last_item.get('question', '') or ''
			last_a_text = last_item.get('answer', '') or ''
		else:
			last_q_text = str(last_item)

	# 1. Use the intelligent classifier in LLMService
	if hasattr(llm_service, 'classify_is_technical'):
		allowed, confidence, reason = await llm_service.classify_is_technical(last_q_text, last_a_text or "", api_key=api_key)
		if allowed and confidence > 0.6:
			return True, f"AI detected: {reason}"

	problem_text = (problem or "").strip() or last_q_text
	if not problem_text:
		return False, "No problem text or recent question available"

	try:
		# Use existing heuristics in LLMService fallback
		is_algo = getattr(llm_service, '_is_algorithm_question', None)
		is_system = getattr(llm_service, '_is_system_design_question', None)
		
		if is_algo and is_algo(problem_text):
			return True, "Detected algorithm/data-structure question"
		if is_system and is_system(problem_text):
			return True, "Detected system-design question"

		# Simple semantic check if LLM call is unavailable
		lower = problem_text.lower()
		technical_kws = [
			'code', 'implement', 'design', 'architecture', 'solve', 
			'program', 'script', 'function', 'method', 'class', 
			'logic', 'algorithm', 'prime', 'number', 'system', 
			'database', 'query', 'sql', 'python', 'java', 'javascript'
		]
		if any(kw in lower for kw in technical_kws):
			return True, "Semantic match for technical query"

		return False, "No technical indicators found"

	except Exception:
		return False, "Error during technical classification"






@router.get("/evaluate/allowed")
async def evaluate_allowed(
	session_id: str, 
	problem: Optional[str] = None, 
	request: Request = None,
	x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
	x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
	authorization: Optional[str] = Header(None, alias="Authorization"),
):
	"""Return whether evaluation is allowed for the given session/problem.
	Frontend can call this to enable/disable the Evaluate button.
	"""
	# Extract API keys from headers (same logic as POST /evaluate)
	groq_key = x_api_key
	gemini_key = x_gemini_key
	if not groq_key and authorization:
		if authorization.startswith("Bearer "):
			groq_key = authorization.split(" ", 1)[1] if " " in authorization else authorization
		else:
			groq_key = authorization
	api_key = gemini_key if gemini_key else groq_key

	user_id = get_user_id_from_request(request) if request else "guest_unknown"
	manager = get_session_manager(user_id)
	
	try:
		session_state = await manager.get_required(session_id)
	except KeyError:
		raise HTTPException(status_code=404, detail="Session not found")

	allowed, reason = await _classify_evaluation_allowed(session_state, problem, api_key=api_key)
	return JSONResponse(status_code=200, content={"allowed": allowed, "reason": reason})


@router.post("/evaluate", response_model=EvaluationOut)
async def evaluate(
	payload: EvaluationIn,
	request: Request,
	response: Response,
	x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
	x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
	authorization: Optional[str] = Header(None, alias="Authorization"),
):
	# Extract API keys from headers
	groq_key = x_api_key
	gemini_key = x_gemini_key
	
	# Fallback to Authorization header for Groq if X-API-Key not provided
	if not groq_key and authorization:
		if authorization.startswith("Bearer "):
			groq_key = authorization.split(" ", 1)[1] if " " in authorization else authorization
		else:
			groq_key = authorization

	# Prefer Gemini if provided in Bridge Settings
	api_key = gemini_key if gemini_key else groq_key
	
	# Fallback to environment keys ONLY in development/local mode
	if not api_key:
		from app.config import settings
		dev_key = settings.gemini_api_key or settings.groq_api_key
		if dev_key:
			api_key = dev_key
			# logger.info("🔧 [DEV MODE] Using environment API key for evaluation")
		else:
			raise HTTPException(
				status_code=401,
				detail="API key required. Please add your Groq or Gemini API key in Bridge Settings."
			)

	# Get user-spec manager
	user_id = get_user_id_from_request(request) or "guest_unknown"
	manager = get_session_manager(user_id)
	
	try:
		await manager.get_required(payload.session_id)
	except KeyError:
		raise HTTPException(status_code=404, detail="Session not found. Create one via POST /api/session and reuse its session_id.")

	if not payload.code.strip():
		raise HTTPException(status_code=400, detail="Empty code")

	# Get session context for cache key
	session_state = await manager.get_required(payload.session_id)

	# Uses AI-powered robust classification
	allowed, reason = await _classify_evaluation_allowed(session_state, payload.problem, payload.code, api_key=api_key)
	
	# ROBUST FALLBACK: If prompt detection fails, check the actual code content!
	# If the user provided actual code structure, we should allow evaluation.
	if not allowed and payload.code:
		code_lower = payload.code.lower()
		from app.config import settings
		code_indicators = list(getattr(settings, "evaluation_code_indicators", None) or [])
		# Check for at least 2 structural indicators to avoid false positives with plain text
		indicators_found = [ki for ki in code_indicators if ki in code_lower]
		if len(indicators_found) >= 2:
			allowed = True
			reason = f"Detected code structure ({', '.join(indicators_found[:3])}...)"

	if not allowed:
		raise HTTPException(status_code=400, detail=f"Evaluation is allowed only for technical, coding, or system-design questions. ({reason})")
	
	# Create cache key based on session + conversation context + code
	# This ensures same code in different conversations gets different evaluations
	conversation_context = ""
	if session_state.qna:
		# Use last 2 QnA pairs for context
		recent_qna = session_state.qna[-2:] if len(session_state.qna) >= 2 else session_state.qna
		for item in recent_qna:
			conversation_context += f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}\n"
	
	cache_key = hashlib.md5(
		f"{payload.session_id}|{conversation_context}|{payload.code.strip()}|{payload.problem or ''}|{payload.language or 'python'}".encode()
	).hexdigest()

	# Check cache first
	cached_result = await _cache_get(cache_key)
	if cached_result is not None:
		# Update session_id to match current request
		cached_result.session_id = payload.session_id

		# Usage tracking (cache hit still counts as an evaluation call)
		with get_db_context() as db:
			track_api_usage(
				db,
				getattr(request.state, "user", None),
				feature="evaluation",
				endpoint=str(request.url.path),
				metadata={"cached": True, "language": payload.language or "python"},
				guest_user_id=user_id if user_id.startswith("guest_") else None,
			)
		
		# Log cache hit
		await auditor.log({
			"type": "evaluation",
			"session_id": payload.session_id,
			"problem": payload.problem,
			"language": payload.language,
			"cached": True,  # This is a cached result
		})
		
		return cached_result

	# Run evaluation (static + LLM critique)
	try:
		critique_text, static = await evaluate_code(payload.problem, payload.code, payload.language or "python", conversation_context, api_key=api_key)
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"LLM evaluation failed: {str(e)}")

	# Extract scores JSON from critique text
	# Look for 'Scores: {...}'
	scores_dict = {
		"correctness": 0.0,
		"optimization": 0.0,
		"approach_explanation": 0.0,
		"complexity_discussion": 0.0,
		"edge_cases_testing": 0.0,
		"total": 0.0,
	}
	try:
		marker = "Scores:"
		if marker in critique_text:
			json_part = critique_text.split(marker, 1)[1].strip()
			# Grab first JSON object
			start = json_part.find("{")
			end = json_part.find("}")
			if start != -1 and end != -1 and end > start:
				blob = json_part[start:end+1]
				import json as _json
				parsed = _json.loads(blob)
				if isinstance(parsed, dict):
					for k, v in parsed.items():
						key = k.lower().strip()
						if key in scores_dict:
							try:
								scores_dict[key] = float(v)
							except (ValueError, TypeError):
								pass
	except Exception:
		pass
	
	# Fallback: Regex extraction if JSON parsing failed
	if scores_dict["total"] == 0.0:
		import re
		try:
			# Look for individual scores in text: "Correctness: 8/10" or "Correctness: 8"
			patterns = {
				"correctness": r"correctness.*?(\d+(?:\.\d+)?)",
				"optimization": r"optimization.*?(\d+(?:\.\d+)?)",
				"approach_explanation": r"(?:approach|explanation).*?(\d+(?:\.\d+)?)",
				"complexity_discussion": r"complexity.*?(\d+(?:\.\d+)?)",
				"edge_cases_testing": r"(?:edge|testing).*?(\d+(?:\.\d+)?)",
				"total": r"total.*?(\d+(?:\.\d+)?)"
			}
			for key, pat in patterns.items():
				m = re.search(pat, critique_text, re.IGNORECASE)
				if m:
					try:
						val = float(m.group(1))
						# Normalize if out of 100
						if val > 10: val /= 10.0
						scores_dict[key] = min(val, 10.0)
					except ValueError:
						pass
		except Exception:
			pass

	# Normalize scores to ensure they fit within 0-10 range
	for key in scores_dict:
		val = scores_dict[key]
		if isinstance(val, (int, float)):
			# If the model thought it was out of 100, scale down
			if val > 10:
				val = val / 10.0
			# Hard clamp to 10.0 to prevent validation errors
			scores_dict[key] = min(float(val), 10.0)

	# Basic section parsing with robust header detection
	def _section(title: str) -> str:
		import re
		# Flexible pattern to match section headers:
		# 1. "Header:"
		# 2. "**Header**" or "**Header:**" or "**Header**:"
		# 3. "### Header" or "## Header"
		# 4. "Header" (at start of line)
		
		# Locate the start of the section
		# We search for the title explicitly to avoid false positives
		candidates = [
			f"{title}:",
			f"**{title}**",
			f"### {title}",
			f"## {title}",
			f"# {title}",
		]
		
		start_idx = -1
		used_marker = ""
		
		for candidate in candidates:
			idx = critique_text.find(candidate)
			if idx != -1:
				if start_idx == -1 or idx < start_idx:
					start_idx = idx
					used_marker = candidate
		
		if start_idx == -1:
			return ""
			
		# Content starts after the marker
		content_start = start_idx + len(used_marker)
		rem = critique_text[content_start:]
		
		# Stop at the next major section
		# We scan for any of the other standard headers
		next_headers = ["Strengths", "Weaknesses", "Recommendations", "Scores", "Summary", "Approach"]
		stop_indices = []
		
		for hdr in next_headers:
			if hdr == title: continue
			
			# Check for various formats of the next header
			for prefix in ["\n", "\n\n", "\n### ", "\n## ", "\n**"]:
				check = f"{prefix}{hdr}"
				idx = rem.find(check)
				if idx != -1:
					stop_indices.append(idx)
					
			# Also check simpler "Header:" if explicitly on a newline
			check_simple = f"\n{hdr}:"
			idx = rem.find(check_simple)
			if idx != -1:
				stop_indices.append(idx)

		if stop_indices:
			# Stop at the earliest next header
			stop_idx = min(stop_indices)
			return rem[:stop_idx].strip()
			
		return rem.strip()

	# Prefer explicit Approach section if present; otherwise fallback to Summary; otherwise use full text
	approach_text = _section("Approach")
	summary = _section("Summary")
	strengths_raw = _section("Strengths")
	weaknesses_raw = _section("Weaknesses")
	recs_raw = _section("Recommendations")

	def _bullets(text: str) -> list[str]:
		items = []
		for line in text.splitlines():
			l = line.strip()
			if l.startswith("- "):
				items.append(l[2:].strip())
		return items

	# Choose the best available approach content
	_best_approach = (approach_text or summary or critique_text).strip()

	resp = EvaluationOut(
		session_id=payload.session_id,
		problem=payload.problem,
		language=(payload.language or "python"),
		approach_auto_explanation=_best_approach,
		feedback_summary=_best_approach,
		strengths=_bullets(strengths_raw),
		weaknesses=_bullets(weaknesses_raw),
		scores=EvaluationScores(**scores_dict),
		static_signals=StaticSignals(**static),
		recommendations=_bullets(recs_raw),
		created_at=utcnow(),
		markdown=f"""
### Approach

{_best_approach}
""",
	)

    # Diagrammatic evaluation disabled temporarily per user request.

	# Cache the result for future requests
	await _cache_set(cache_key, resp)

	# Usage tracking (Phase 2 billing/analytics)
	with get_db_context() as db:
		track_api_usage(
			db,
			getattr(request.state, "user", None),
			feature="evaluation",
			endpoint=str(request.url.path),
			metadata={"cached": False, "language": payload.language or "python"},
			guest_user_id=user_id if user_id.startswith("guest_") else None,
		)



	await auditor.log({
		"type": "evaluation",
		"session_id": payload.session_id,
		"problem": payload.problem,
		"language": payload.language,
		"scores": scores_dict,
		"cached": False,  # This is a new evaluation
	})

	return resp


