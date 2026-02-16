"""
Practice Mode configuration helper.
Builds PracticeModeConfig from main settings.
"""

from app.schemas import PracticeModeConfig, TTSConfig, STTConfig, SpeechAnalyticsConfig
from app.config import settings


def get_practice_config() -> PracticeModeConfig:
	"""Get Practice Mode configuration from main settings."""
	return PracticeModeConfig(
		tts=TTSConfig(
			engine=settings.practice_tts_engine,
			tts_model_name=settings.practice_tts_model,
			sample_rate=22050,
			max_generation_time=settings.practice_tts_timeout_seconds
		),
		stt=STTConfig(
			stt_model_size=settings.practice_stt_model_size,
			device=settings.practice_stt_device,
			compute_type="int8",
			max_transcription_time=settings.practice_stt_max_transcription_time_seconds,
			target_rtf=settings.practice_stt_target_rtf,
		),
		analytics=SpeechAnalyticsConfig(),
		audio_storage_path=settings.practice_audio_storage,
		session_timeout_minutes=settings.practice_session_timeout,
		max_concurrent_sessions=settings.practice_max_sessions
	)


# Export default config
default_config = get_practice_config()
