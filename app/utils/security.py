from __future__ import annotations

from fastapi import Header, HTTPException, status
from typing import Optional

from app.config import settings


async def verify_api_key(authorization: Optional[str] = Header(default=None)) -> None:
	if not settings.api_key:
		return
	if not authorization or not authorization.startswith("Bearer "):
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")
	key = authorization.removeprefix("Bearer ")
	if key != settings.api_key:
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


async def websocket_verify_api_key(sec_websocket_protocol: Optional[str] = Header(default=None)) -> None:
	# Expect API key via Sec-WebSocket-Protocol when provided
	if not settings.api_key:
		return
	if not sec_websocket_protocol:
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")
	if sec_websocket_protocol != settings.api_key:
		raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
