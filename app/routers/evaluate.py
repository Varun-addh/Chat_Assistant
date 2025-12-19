from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, Header
from fastapi.responses import JSONResponse
from datetime import datetime
import json
import hashlib
from typing import Dict, Optional

from app.schemas import EvaluationIn, EvaluationOut, EvaluationScores, StaticSignals
from app.services.session_manager import session_manager
from app.services.code_evaluation_service import evaluate_code
from app.utils.audit import auditor
from app.services.llm_service import llm_service


router = APIRouter()

# In-memory cache for evaluations
_evaluation_cache: Dict[str, EvaluationOut] = {}


def _classify_evaluation_allowed(session_state, problem: Optional[str]) -> tuple[bool, str]:
	"""Return (allowed, reason) whether evaluation should be permitted.
	Uses llm_service heuristics when available, with conservative fallbacks.
	"""
	recent_qna = session_state.qna[-2:] if session_state.qna else []
	last_q_text = ""
	if recent_qna:
		last_item = recent_qna[-1]
		if isinstance(last_item, dict):
			last_q_text = last_item.get('question', '') or ''
		else:
			last_q_text = str(last_item)

	problem_text = (problem or "").strip() or last_q_text
	if not problem_text:
		return False, "No problem text or recent question available"

	try:
		is_algo = getattr(llm_service, '_is_algorithm_question', None)
		is_system = getattr(llm_service, '_is_system_design_question', None)
		is_tech_strategy = getattr(llm_service, '_is_technical_strategy_question', None)
		allowed = False
		pt = problem_text.strip()
		if is_algo and is_algo(pt):
			return True, "Detected algorithm/data-structure question"
		if is_system and is_system(pt):
			return True, "Detected system-design question"
		if is_tech_strategy and is_tech_strategy(pt):
			return True, "Detected technical/strategy question"

		# Fallback indicators for code/implementation
		lower = problem_text.lower()
		code_indicators = ['implement', 'write code', 'solve', 'function', 'class', 'algorithm', 'data structure', 'sql', 'query']
		if any(ind in lower for ind in code_indicators):
			return True, "Problem text contains code/implementation indicators"

		return False, "No technical/code indicators found"

	except Exception:
		lower = problem_text.lower()
		if any(k in lower for k in ['code', 'implement', 'algorithm', 'system design', 'design', 'sql']):
			return True, "Fallback detected code-related keywords"
		return False, "Fallback: no code-related keywords found"






@router.get("/evaluate/allowed")
async def evaluate_allowed(session_id: str, problem: Optional[str] = None):
	"""Return whether evaluation is allowed for the given session/problem.
	Frontend can call this to enable/disable the Evaluate button.
	"""
	try:
		session_state = await session_manager.get_required(session_id)
	except KeyError:
		raise HTTPException(status_code=404, detail="Session not found")

	allowed, reason = _classify_evaluation_allowed(session_state, problem)
	return JSONResponse(status_code=200, content={"allowed": allowed, "reason": reason})


@router.post("/evaluate", response_model=EvaluationOut)
async def evaluate(
	payload: EvaluationIn,
	request: Request,
	response: Response,
	x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
	authorization: Optional[str] = Header(None, alias="Authorization"),
):
	# Extract API key
	api_key = x_api_key
	if not api_key and authorization:
		if authorization.startswith("Bearer "):
			api_key = authorization.split(" ")[1]

	try:
		await session_manager.get_required(payload.session_id)
	except KeyError:
		raise HTTPException(status_code=404, detail="Session not found. Create one via POST /api/session and reuse its session_id.")

	if not payload.code.strip():
		raise HTTPException(status_code=400, detail="Empty code")

	# Get session context for cache key
	session_state = await session_manager.get_required(payload.session_id)

	# Use centralized classifier to decide if evaluation is allowed
	allowed, reason = _classify_evaluation_allowed(session_state, payload.problem)
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
	if cache_key in _evaluation_cache:
		cached_result = _evaluation_cache[cache_key]
		# Update session_id to match current request
		cached_result.session_id = payload.session_id
		
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
				scores_dict.update(json.loads(blob))
	except Exception:
		pass

	# Basic section parsing
	def _section(title: str) -> str:
		key = title + ":"
		if key not in critique_text:
			return ""
		rem = critique_text.split(key, 1)[1]
		# stop at next heading
		for h in ["\n\nStrengths:", "\n\nWeaknesses:", "\n\nScores:", "\n\nRecommendations:"]:
			if h in rem:
				rem = rem.split(h, 1)[0]
				break
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
		created_at=datetime.utcnow(),
		markdown=f"""
### Approach

{_best_approach}
""",
	)

    # Diagrammatic evaluation disabled temporarily per user request.

	# Cache the result for future requests
	_evaluation_cache[cache_key] = resp



	await auditor.log({
		"type": "evaluation",
		"session_id": payload.session_id,
		"problem": payload.problem,
		"language": payload.language,
		"scores": scores_dict,
		"cached": False,  # This is a new evaluation
	})

	return resp


