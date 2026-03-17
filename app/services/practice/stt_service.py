from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncIterator, Optional

from app.config import settings


class STTService:
	def __init__(self) -> None:
		self._provider = (settings.stt_provider or "none").strip().lower()
		self._enabled = self._provider == "whisper"
		self._local_whisper_service: Optional[object] = None

	@property
	def enabled(self) -> bool:
		return self._enabled

	def _get_local_whisper_service(self):
		if self._provider != "whisper":
			raise RuntimeError(
				f"WebSocket STT is not configured. Set STT_PROVIDER=whisper to enable real transcription; current provider is '{self._provider}'."
			)

		if self._local_whisper_service is None:
			from app.config_practice_mode import get_practice_config
			from app.services.practice.local_stt_service import LocalSTTService

			self._local_whisper_service = LocalSTTService(get_practice_config().stt)

		return self._local_whisper_service

	async def stream_transcribe(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
		if self._provider == "none":
			raise RuntimeError("WebSocket STT is disabled. Configure STT_PROVIDER=whisper to enable transcription.")
		if self._provider != "whisper":
			raise RuntimeError(
				f"WebSocket STT provider '{self._provider}' is not implemented. Use STT_PROVIDER=whisper or disable the endpoint."
			)

		audio_chunks: list[bytes] = []
		async for chunk in audio_stream:
			if chunk:
				audio_chunks.append(chunk)

		if not audio_chunks:
			return

		with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
			for chunk in audio_chunks:
				temp_file.write(chunk)
			temp_path = temp_file.name

		try:
			whisper_service = await asyncio.to_thread(self._get_local_whisper_service)
			transcript, _ = await whisper_service.transcribe_async(temp_path)
			text = (transcript or "").strip()
			if text:
				yield text
		finally:
			Path(temp_path).unlink(missing_ok=True)


stt_service = STTService()
