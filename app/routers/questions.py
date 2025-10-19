from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from datetime import datetime

from app.schemas import CreateSessionResponse, QuestionIn, AnswerOut, SessionHistory, QnA, SessionList, SessionSummary
from app.services.session_manager import session_manager
from app.services.llm_service import llm_service
from app.services.code_evaluation_service import evaluate_code
from app.utils.security import verify_api_key
from app.utils.audit import auditor
import asyncio


router = APIRouter()


async def _auto_evaluate_if_code(session_id: str, question: str, answer: str) -> None:
	"""Auto-evaluate if the answer contains code blocks."""
	import re
	
	# Look for code blocks in the answer
	code_pattern = r'```(\w+)?\n(.*?)\n```'
	matches = re.findall(code_pattern, answer, re.DOTALL)
	
	if not matches:
		return
	
	# Get the most substantial code block
	best_code = ""
	best_lang = "python"
	for lang, code in matches:
		if len(code.strip()) > len(best_code.strip()):
			best_code = code.strip()
			best_lang = lang or "python"
	
	if not best_code:
		return
	
	# Auto-trigger evaluation in background
	try:
		# Get conversation context
		session_state = await session_manager.get_required(session_id)
		conversation_context = ""
		if session_state.qna:
			recent_qna = session_state.qna[-2:] if len(session_state.qna) >= 2 else session_state.qna
			for item in recent_qna:
				conversation_context += f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}\n"
		
		# Run evaluation
		await evaluate_code(question, best_code, best_lang, conversation_context)
		
		# Log auto-evaluation
		await auditor.log({
			"type": "auto_evaluation",
			"session_id": session_id,
			"question": question,
			"language": best_lang,
			"auto_triggered": True,
		})
	except Exception as e:
		# Don't fail the main request if evaluation fails
		await auditor.log({
			"type": "auto_evaluation_error",
			"session_id": session_id,
			"error": str(e),
		})


@router.post("/session", response_model=CreateSessionResponse)
async def create_session():
	state = await session_manager.create_session()
	return CreateSessionResponse(session_id=state.session_id)


@router.post("/question")
async def submit_question(payload: QuestionIn):
	try:
		state = await session_manager.get_required(payload.session_id)
	except KeyError:
		raise HTTPException(status_code=404, detail="Session not found. Create one via POST /api/session and reuse its session_id.")

	if not payload.question.strip():
		raise HTTPException(status_code=400, detail="Empty question")

	# Read any stored profile for this session
	profile_text = state.profile_text

	if payload.stream:
		async def event_gen():
			collected: list[str] = []
			# Provide recent QnA as context for follow-ups
			previous_qna = state.qna
			async for chunk in llm_service.stream_answer(
				payload.question,
				payload.system_prompt,
				profile_text=profile_text,
				previous_qna=previous_qna,
				style_mode=payload.style_mode,
				tone=payload.tone,
				layout=payload.layout,
				variability=payload.variability,
				seed=payload.seed,
			):
				collected.append(chunk)
				yield f"data: {chunk}\n\n"
			# On stream end, persist the full answer
			full_answer = "".join(collected)
			await session_manager.append_qna(state.session_id, payload.question, full_answer)
			await auditor.log({
				"type": "qna",
				"session_id": state.session_id,
				"question": payload.question,
				"answer": full_answer,
			})
			
			# Auto-evaluate if response contains code
			asyncio.create_task(_auto_evaluate_if_code(state.session_id, payload.question, full_answer))
			
			yield "event: end\n\n"

		return StreamingResponse(event_gen(), media_type="text/event-stream")

	# Provide recent QnA as context for follow-ups
	previous_qna = state.qna
	answer = await llm_service.generate_answer(
		payload.question,
		payload.system_prompt,
		profile_text=profile_text,
		previous_qna=previous_qna,
		style_mode=payload.style_mode,
		tone=payload.tone,
		layout=payload.layout,
		variability=payload.variability,
		seed=payload.seed,
	)
	await session_manager.append_qna(state.session_id, payload.question, answer)
	await auditor.log({
		"type": "qna",
		"session_id": state.session_id,
		"question": payload.question,
		"answer": answer,
	})
	
	# Auto-evaluate if response contains code
	asyncio.create_task(_auto_evaluate_if_code(state.session_id, payload.question, answer))
	
	return AnswerOut(answer=answer, created_at=datetime.utcnow())


@router.post("/upload_profile")
async def upload_profile(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
	# Ensure session exists
	try:
		await session_manager.get_required(session_id)
	except KeyError:
		raise HTTPException(status_code=404, detail="Session not found. Create one via POST /api/session and reuse its session_id.")

	# Determine how to read the file; support text and pdf minimally
	filename = (file.filename or "").lower()
	content_type = (file.content_type or "").lower()

	# Read bytes
	data = await file.read()

	text: str = ""
	try:
		if filename.endswith(".txt") or content_type.startswith("text/"):
			text = data.decode("utf-8", errors="ignore")
		elif filename.endswith(".md"):
			text = data.decode("utf-8", errors="ignore")
		elif filename.endswith(".pdf") or content_type == "application/pdf":
			# Lazy import to avoid heavy dependency at startup
			try:
				from app.utils.text_extract import extract_text_from_pdf  # type: ignore
			except Exception:
				raise HTTPException(status_code=415, detail="PDF support not available. Please install pdfminer.six or upload a text/markdown file.")
			text = extract_text_from_pdf(data)
		else:
			# Fallback: try utf-8
			text = data.decode("utf-8", errors="ignore")
	except UnicodeDecodeError:
		raise HTTPException(status_code=415, detail="Unable to decode file. Please upload a UTF-8 text, markdown, or PDF file.")

	if not text.strip():
		raise HTTPException(status_code=400, detail="Uploaded file appears empty.")

	await session_manager.set_profile_text(session_id, text)
	await auditor.log({
		"type": "profile_upload",
		"session_id": session_id,
		"filename": file.filename,
		"bytes": len(data),
	})

	return {"status": "ok", "characters": len(text)}


@router.get("/history/{session_id}", response_model=SessionHistory)
async def get_history(session_id: str):
	try:
		state = await session_manager.get_required(session_id)
	except KeyError:
		raise HTTPException(status_code=404, detail="Session not found")
	items = [
		QnA(question=i["question"], answer=i["answer"], created_at=datetime.fromisoformat(i["created_at"]))
		for i in state.qna
	]
	return SessionHistory(session_id=session_id, items=items)


@router.get("/sessions", response_model=SessionList)
async def list_sessions():
	items_raw = await session_manager.list_sessions()
	items = [
		SessionSummary(
			session_id=i["session_id"],
			last_update=datetime.fromisoformat(i["last_update"]),
			qna_count=i["qna_count"],
		)
		for i in items_raw
	]
	return SessionList(items=items)


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
	deleted = await session_manager.delete_session(session_id)
	if not deleted:
		raise HTTPException(status_code=404, detail="Session not found")
	return {"status": "ok", "deleted": True}


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
	try:
		await session_manager.clear_history(session_id)
	except KeyError:
		raise HTTPException(status_code=404, detail="Session not found")
	return {"status": "ok"}


@router.delete("/history/{session_id}/{index}")
async def delete_qna_item(session_id: str, index: int):
	try:
		await session_manager.remove_qna(session_id, index)
	except KeyError:
		raise HTTPException(status_code=404, detail="Session not found")
	except IndexError:
		raise HTTPException(status_code=400, detail="QnA index out of range")
	return {"status": "ok"}
