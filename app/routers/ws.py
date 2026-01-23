from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import AsyncIterator

from app.services.core.session_manager import get_session_manager
from app.services.practice.stt_service import stt_service
from app.utils.security import websocket_verify_api_key


router = APIRouter()


async def _iter_audio(websocket: WebSocket) -> AsyncIterator[bytes]:
	while True:
		msg = await websocket.receive()
		if "bytes" in msg and msg["bytes"] is not None:
			yield msg["bytes"]
		elif msg.get("text") == "__end__":
			break


@router.websocket("/ws/stt/{session_id}")
async def ws_stt(websocket: WebSocket, session_id: str, _: None = Depends(websocket_verify_api_key)):
	await websocket.accept()
	
	# Get user_id from query params or default
	user_id = websocket.query_params.get("user_id", "default")
	manager = get_session_manager(user_id)
	
	# Ensure session exists
	await manager.get_required(session_id)

	try:
		async for text in stt_service.stream_transcribe(_iter_audio(websocket)):
			await manager.append_partial_transcript(session_id, text)
			await websocket.send_json({
				"type": "partial_transcript",
				"text": text,
			})
		await websocket.send_json({"type": "end"})
	except WebSocketDisconnect:
		return
