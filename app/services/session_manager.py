from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional
import uuid
import asyncio
import json
import os
from pathlib import Path


@dataclass
class SessionState:
	session_id: str
	qna: List[dict] = field(default_factory=list)
	partial_transcript: str = ""
	last_update: datetime = field(default_factory=datetime.utcnow)
	profile_text: str = ""


class SessionManager:
	def __init__(self) -> None:
		self._sessions: Dict[str, SessionState] = {}
		self._lock = asyncio.Lock()
		# Persistence directory
		self._data_dir = Path("data") / "sessions"
		self._data_dir.mkdir(parents=True, exist_ok=True)
		self._load_all()

	def _session_path(self, session_id: str) -> Path:
		return self._data_dir / f"{session_id}.json"

	def _serialize(self, state: SessionState) -> dict:
		data = asdict(state)
		# Convert datetime to isoformat
		if isinstance(state.last_update, datetime):
			data["last_update"] = state.last_update.isoformat()
		return data

	def _deserialize(self, data: dict) -> SessionState:
		last_update = data.get("last_update")
		if isinstance(last_update, str):
			try:
				last_dt = datetime.fromisoformat(last_update)
			except Exception:
				last_dt = datetime.utcnow()
		else:
			last_dt = datetime.utcnow()
		return SessionState(
			session_id=data["session_id"],
			qna=list(data.get("qna", [])),
			partial_transcript=data.get("partial_transcript", ""),
			last_update=last_dt,
			profile_text=data.get("profile_text", ""),
		)

	def _load_all(self) -> None:
		try:
			for p in self._data_dir.glob("*.json"):
				with p.open("r", encoding="utf-8") as f:
					raw = json.load(f)
					state = self._deserialize(raw)
					self._sessions[state.session_id] = state
		except Exception:
			# Best-effort; ignore corrupt files
			pass

	def _save(self, state: SessionState) -> None:
		path = self._session_path(state.session_id)
		try:
			with path.open("w", encoding="utf-8") as f:
				json.dump(self._serialize(state), f, ensure_ascii=False, indent=2)
		except Exception:
			# Best-effort; do not crash app on IO errors
			pass

	async def create_session(self) -> SessionState:
		async with self._lock:
			session_id = str(uuid.uuid4())
			state = SessionState(session_id=session_id)
			self._sessions[session_id] = state
			self._save(state)
			return state

	async def get(self, session_id: str) -> Optional[SessionState]:
		return self._sessions.get(session_id)

	async def append_qna(self, session_id: str, question: str, answer: str) -> None:
		state = await self.get_required(session_id)
		state.qna.append({
			"question": question,
			"answer": answer,
			"created_at": datetime.utcnow().isoformat(),
		})
		state.last_update = datetime.utcnow()
		self._save(state)

	async def set_partial_transcript(self, session_id: str, text: str) -> None:
		state = await self.get_required(session_id)
		state.partial_transcript = text
		state.last_update = datetime.utcnow()
		self._save(state)

	async def append_partial_transcript(self, session_id: str, text: str) -> None:
		state = await self.get_required(session_id)
		if state.partial_transcript and not state.partial_transcript.endswith(" "):
			state.partial_transcript += " "
		state.partial_transcript += text
		state.last_update = datetime.utcnow()
		self._save(state)

	async def get_required(self, session_id: str) -> SessionState:
		state = await self.get(session_id)
		if state is None:
			raise KeyError("session not found")
		return state

	async def set_profile_text(self, session_id: str, text: str) -> None:
		state = await self.get_required(session_id)
		state.profile_text = text.strip()
		state.last_update = datetime.utcnow()
		self._save(state)

	async def get_profile_text(self, session_id: str) -> str:
		state = await self.get_required(session_id)
		return state.profile_text

	async def list_sessions(self) -> List[dict]:
		"""Return lightweight session summaries for frontend lists."""
		items: List[dict] = []
		for s in self._sessions.values():
			items.append({
				"session_id": s.session_id,
				"last_update": s.last_update.isoformat(),
				"qna_count": len(s.qna),
			})
		# Newest first
		items.sort(key=lambda x: x["last_update"], reverse=True)
		return items

	async def delete_session(self, session_id: str) -> bool:
		"""Delete an entire session and its persisted file. Returns True if deleted."""
		async with self._lock:
			state = self._sessions.pop(session_id, None)
			deleted = state is not None
			# Remove file if exists
			path = self._session_path(session_id)
			try:
				if path.exists():
					path.unlink()
			except Exception:
				pass
			return deleted

	async def clear_history(self, session_id: str) -> None:
		"""Clear QnA history for a session but keep the session and profile."""
		state = await self.get_required(session_id)
		state.qna.clear()
		state.last_update = datetime.utcnow()
		self._save(state)

	async def remove_qna(self, session_id: str, index: int) -> None:
		"""Remove a single QnA entry by zero-based index."""
		state = await self.get_required(session_id)
		if index < 0 or index >= len(state.qna):
			raise IndexError("qna index out of range")
		state.qna.pop(index)
		state.last_update = datetime.utcnow()
		self._save(state)


session_manager = SessionManager()
