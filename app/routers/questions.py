from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from datetime import datetime

from app.schemas import CreateSessionResponse, QuestionIn, AnswerOut, SessionHistory, QnA, SessionList, SessionSummary
from app.services.session_manager import session_manager
from app.services.llm_service import llm_service
from app.utils.security import verify_api_key
from app.utils.audit import auditor


router = APIRouter()


@router.post("/session", response_model=CreateSessionResponse)
async def create_session(_: None = Depends(verify_api_key)):
	state = await session_manager.create_session()
	return CreateSessionResponse(session_id=state.session_id)


@router.post("/question")
async def submit_question(payload: QuestionIn, _: None = Depends(verify_api_key)):
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
			async for chunk in llm_service.stream_answer(payload.question, payload.system_prompt, profile_text=profile_text, previous_qna=previous_qna):
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
			yield "event: end\n\n"

		return StreamingResponse(event_gen(), media_type="text/event-stream")

	# Provide recent QnA as context for follow-ups
	previous_qna = state.qna
	answer = await llm_service.generate_answer(payload.question, payload.system_prompt, profile_text=profile_text, previous_qna=previous_qna)
	await session_manager.append_qna(state.session_id, payload.question, answer)
	await auditor.log({
		"type": "qna",
		"session_id": state.session_id,
		"question": payload.question,
		"answer": answer,
	})
	return AnswerOut(answer=answer, created_at=datetime.utcnow())


@router.post("/upload_profile")
async def upload_profile(
	file: UploadFile = File(...),
	session_id: str = Form(...),
	_: None = Depends(verify_api_key),
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
async def get_history(session_id: str, _: None = Depends(verify_api_key)):
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
async def list_sessions(_: None = Depends(verify_api_key)):
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
async def delete_session(session_id: str, _: None = Depends(verify_api_key)):
	deleted = await session_manager.delete_session(session_id)
	if not deleted:
		raise HTTPException(status_code=404, detail="Session not found")
	return {"status": "ok", "deleted": True}


@router.delete("/history/{session_id}")
async def clear_history(session_id: str, _: None = Depends(verify_api_key)):
	try:
		await session_manager.clear_history(session_id)
	except KeyError:
		raise HTTPException(status_code=404, detail="Session not found")
	return {"status": "ok"}


@router.delete("/history/{session_id}/{index}")
async def delete_qna_item(session_id: str, index: int, _: None = Depends(verify_api_key)):
	try:
		await session_manager.remove_qna(session_id, index)
	except KeyError:
		raise HTTPException(status_code=404, detail="Session not found")
	except IndexError:
		raise HTTPException(status_code=400, detail="QnA index out of range")
	return {"status": "ok"}
