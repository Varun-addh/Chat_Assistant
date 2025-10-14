from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from datetime import datetime
import json

from app.schemas import EvaluationIn, EvaluationOut, EvaluationScores, StaticSignals
from app.services.session_manager import session_manager
from app.services.code_evaluation_service import evaluate_code
from app.utils.audit import auditor


router = APIRouter()


@router.options("/evaluate")
async def evaluate_cors_options(request: Request) -> Response:
	origin = request.headers.get("origin", "*")
	acr_headers = request.headers.get("access-control-request-headers", "*")
	headers = {
		"Access-Control-Allow-Origin": origin,
		"Vary": "Origin",
		"Access-Control-Allow-Headers": acr_headers,
		"Access-Control-Allow-Methods": "POST, OPTIONS",
		"Access-Control-Max-Age": "3600",
	}
	return Response(status_code=204, headers=headers)


@router.post("/evaluate", response_model=EvaluationOut)
async def evaluate(payload: EvaluationIn, request: Request, response: Response):
	try:
		await session_manager.get_required(payload.session_id)
	except KeyError:
		raise HTTPException(status_code=404, detail="Session not found. Create one via POST /api/session and reuse its session_id.")

	if not payload.code.strip():
		raise HTTPException(status_code=400, detail="Empty code")

	# Run evaluation (static + LLM critique)
	try:
		critique_text, static = await evaluate_code(payload.problem, payload.code, payload.language or "python")
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

	resp = EvaluationOut(
		session_id=payload.session_id,
		problem=payload.problem,
		language=(payload.language or "python"),
		approach_auto_explanation=summary or "",
		feedback_summary=summary or "",
		strengths=_bullets(strengths_raw),
		weaknesses=_bullets(weaknesses_raw),
		scores=EvaluationScores(**scores_dict),
		static_signals=StaticSignals(**static),
		recommendations=_bullets(recs_raw),
		created_at=datetime.utcnow(),
	)

	# Ensure CORS header mirrors other endpoints for some hosts that require explicit setting
	origin = request.headers.get("origin")
	if origin:
		response.headers["Access-Control-Allow-Origin"] = origin
		response.headers["Vary"] = "Origin"

	await auditor.log({
		"type": "evaluation",
		"session_id": payload.session_id,
		"problem": payload.problem,
		"language": payload.language,
		"scores": scores_dict,
	})

	return resp


