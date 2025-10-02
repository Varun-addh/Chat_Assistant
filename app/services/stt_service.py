from __future__ import annotations

from typing import AsyncIterator, Optional

from app.config import settings


class STTService:
	def __init__(self) -> None:
		self._provider = settings.stt_provider
		self._enabled = self._provider != "none"

	@property
	def enabled(self) -> bool:
		return self._enabled

	async def stream_transcribe(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
		if not self._enabled:
			# Echo fake words per chunk for local testing without a provider
			async for _ in audio_stream:
				yield "(audio)"
			return

		# Placeholder for real provider integration (OpenAI Realtime/Whisper, Deepgram, etc.)
		async for _ in audio_stream:
			# In a real integration, send bytes to provider and yield partial transcripts
			yield "..."


stt_service = STTService()
