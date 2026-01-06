from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import uuid
import asyncio
import json
import os
from pathlib import Path
import time


@dataclass
class SessionState:
	session_id: str
	qna: List[dict] = field(default_factory=list)
	partial_transcript: str = ""
	last_update: datetime = field(default_factory=datetime.utcnow)
	profile_text: str = ""
	custom_title: Optional[str] = None


class SessionManager:
	def __init__(self, user_id: str = "default") -> None:
		self.user_id = user_id
		self._sessions: Dict[str, SessionState] = {}
		self._lock = asyncio.Lock()
		# Persistence directory - User isolated
		self._data_dir = Path("data/sessions") / user_id
		self._data_dir.mkdir(parents=True, exist_ok=True)
		self._load_all()
		# 🛡️ Debounce protection: Track last session creation time
		self._last_session_creation: Optional[float] = None
		self._last_created_session_id: Optional[str] = None
		self._debounce_window_seconds: float = 1.0  # Prevent duplicate sessions within 1 second

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
			custom_title=data.get("custom_title"),
		)

	def _load_all(self) -> None:
		"""Load all persisted sessions for THIS user and sync with disk."""
		if not self._data_dir.exists():
			self._sessions.clear()
			return
			
		found_ids = set()
		loaded_count = 0
		for p in self._data_dir.glob("*.json"):
			# Skip non-session files (like mock_interview_sessions.json)
			# Session files are always {uuid}.json
			if "-" not in p.stem or len(p.stem) < 30:
				continue
				
			session_id = p.stem
			found_ids.add(session_id)
			
			# If already in memory, we skip loading to avoid overwriting recent in-memory changes.
			# In a multi-worker setup, we prioritize disk presence over stale memory.
			if session_id not in self._sessions:
				try:
					with p.open("r", encoding="utf-8") as f:
						raw = json.load(f)
						state = self._deserialize(raw)
						self._sessions[state.session_id] = state
						loaded_count += 1
				except Exception:
					continue
		
		# Sync: Remove in-memory sessions that are no longer on disk
		to_remove = [sid for sid in self._sessions if sid not in found_ids]
		for sid in to_remove:
			del self._sessions[sid]
		
		if loaded_count > 0 or to_remove:
			import logging
			logging.getLogger(__name__).info(f"🔄 Session cache synced for {self.user_id}: +{loaded_count} new, -{len(to_remove)} deleted")

	def _save(self, state: SessionState) -> None:
		path = self._session_path(state.session_id)
		try:
			# Ensure directory exists before saving
			path.parent.mkdir(parents=True, exist_ok=True)
			with path.open("w", encoding="utf-8") as f:
				json.dump(self._serialize(state), f, ensure_ascii=False, indent=2)
				f.flush()  # Explicitly flush to disk
				os.fsync(f.fileno())  # Force OS to write to disk
			print(f"✅ Saved session {state.session_id} for user {self.user_id} ({len(state.qna)} Q&A pairs)")
		except Exception as e:
			print(f"❌ Failed to save session {state.session_id} for user {self.user_id}: {e}")

	async def create_session(self) -> SessionState:
		async with self._lock:
			# �️ Debounce protection: Prevent rapid duplicate session creation
			current_time = time.time()
			
			if (
				self._last_session_creation is not None
				and self._last_created_session_id is not None
				and (current_time - self._last_session_creation) < self._debounce_window_seconds
			):
				# Return the recently created session instead of creating a new one
				recent_session = self._sessions.get(self._last_created_session_id)
				if recent_session:
					print(f"🛡️ Debounce: Prevented duplicate session creation. Returning session {self._last_created_session_id}")
					return recent_session
			
			# 🚀 Optimization: Reuse existing empty session if possible to prevent sidebar spam
			for state in self._sessions.values():
				if not state.qna and not state.custom_title and not state.profile_text:
					# Update timestamp so it appears at the top
					state.last_update = datetime.utcnow()
					# Update debounce tracking
					self._last_session_creation = current_time
					self._last_created_session_id = state.session_id
					return state

			# Create new session
			session_id = str(uuid.uuid4())
			state = SessionState(session_id=session_id)
			self._sessions[session_id] = state
			self._save(state)
			
			# Update debounce tracking
			self._last_session_creation = current_time
			self._last_created_session_id = session_id
			
			return state

	async def get(self, session_id: str) -> Optional[SessionState]:
		# Check in-memory cache first
		state = self._sessions.get(session_id)
		if state:
			return state
		
		# If not in memory, try loading from disk (multi-worker scenario)
		path = self._session_path(session_id)
		if path.exists():
			try:
				with path.open("r", encoding="utf-8") as f:
					raw = json.load(f)
					state = self._deserialize(raw)
					self._sessions[session_id] = state  # Cache it
					print(f"📥 Loaded session {session_id} from disk for user {self.user_id}")
					return state
			except Exception as e:
				print(f"⚠️ Failed to load session {session_id} from disk: {e}")
		
		return None

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
		if not state:
			raise KeyError(f"Session {session_id} not found")
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
		# Reload from disk first to catch sessions created/updated by other workers
		self._load_all()
		
		items: List[dict] = []
		for s in self._sessions.values():
			# Determine title: Custom title > First question > "New Chat"
			title = s.custom_title
			if not title:
				title = "New Chat"
				if s.qna and len(s.qna) > 0:
					first_q = s.qna[0].get("question", "")
					title = (first_q[:50] + "...") if len(first_q) > 50 else first_q
			
			# 🧹 Cleanup: Skip old empty sessions to keep the sidebar clean
			has_content = (s.qna and len(s.qna) > 0) or s.custom_title or s.profile_text
			
			# We only show empty sessions if they are very recent (e.g. just created by the user)
			# or if they have actual content.
			is_recent = (datetime.utcnow() - s.last_update).total_seconds() < 300 # 5 minutes
			
			if not has_content and not is_recent:
				continue

			items.append({
				"session_id": s.session_id,
				"title": title,
				"last_update": s.last_update.isoformat(),
				"qna_count": len(s.qna),
			})
		# Newest first
		items.sort(key=lambda x: x["last_update"], reverse=True)
		return items

	async def set_session_title(self, session_id: str, title: str) -> None:
		"""Set a custom title for the session."""
		state = await self.get_required(session_id)
		state.custom_title = title.strip()
		state.last_update = datetime.utcnow()
		self._save(state)

	async def delete_session(self, session_id: str) -> bool:
		"""Delete an entire session and its persisted file. Returns True if deleted."""
		async with self._lock:
			state = self._sessions.pop(session_id, None)
			in_memory_deleted = state is not None
			
			# Remove file from disk
			path = self._session_path(session_id)
			file_deleted = False
			try:
				if path.exists():
					path.unlink()
					file_deleted = True
			except Exception:
				pass
				
			return in_memory_deleted or file_deleted

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


# Cache managers per user to avoid frequent reloading
_user_managers: Dict[str, SessionManager] = {}

def get_session_manager(user_id: str = "default") -> SessionManager:
	if user_id not in _user_managers:
		_user_managers[user_id] = SessionManager(user_id)
	return _user_managers[user_id]
