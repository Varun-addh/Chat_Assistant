from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import List
import secrets
import os
from dotenv import load_dotenv


# Ensure .env is loaded eagerly
load_dotenv(dotenv_path=".env")


class Settings(BaseSettings):
	model_config = SettingsConfigDict(
		env_file=".env",
		env_file_encoding="utf-8",
		extra="ignore",
	)

	# Server
	host: str = "0.0.0.0"
	port: int = 8000
	cors_allow_origins: List[str] = ["*"]

	# Auth
	api_key: str | None = None  # simple bearer key if provided
	cookie_secret: str = secrets.token_urlsafe(32)

	# LLM Provider Selection
	llm_provider: str = "gemini"  # default global provider; Interview Intelligence overrides to Groq internally

	# Cohere
	cohere_api_key: str | None = None

	# Judge0
	judge0_api_key: str | None = None

	# Groq
	groq_api_key: str | None = None
	groq_model: str = "llama-3.3-70b-versatile"
	answer_temperature: float = 1
	groq_top_p: float | None = 1
	groq_max_tokens: int | None = 8000  # Override automatic token limit calculation
	groq_max_tokens_simple: int = 300  # For simple questions
	groq_max_tokens_code: int = 800  # For code questions
	groq_max_tokens_complex: int = 1200  # For complex topics
	groq_reasoning_effort: str | None = None  # e.g., "medium"
	groq_stream: bool = True

	# Google Gemini
	gemini_api_key: str | None = None
	# Update default to Gemini 3 preview model. Can be overridden via env var (GEMINI_MODEL).
	gemini_model: str = "models/gemini-flash-latest"

	# Serper API (for faster, more reliable web search - 2500 free searches/month)
	serper_api_key: str | None = None

	# STT
	stt_provider: str = "none"  # options: none, openai, deepgram, whisper

	# Logging
	log_level: str = "INFO"
	analytics_path: str | None = None  # e.g., logs/qna.jsonl

	# Feature Flags
	enable_hybrid_search: bool = True
	enable_reranking: bool = True
	enable_code_execution: bool = True
	enable_query_expansion: bool = True
	enable_streaming: bool = True

	# Practice Mode Configuration
	practice_mode_enabled: bool = True
	practice_tts_engine: str = "gtts"  # options: pyttsx3 (offline), gtts (online) - gtts recommended to avoid blocking issues
	practice_tts_model: str = "default"  # Not used for pyttsx3/gtts
	practice_stt_model_size: str = "base"  # options: tiny, base, small, medium, large
	practice_stt_device: str = "cpu"  # options: cpu, cuda
	practice_audio_storage: str = "data/practice_audio"
	practice_max_sessions: int = 100
	practice_session_timeout: int = 30  # minutes

	# Interview Configuration
	default_question_count: int = 5  # Default number of questions if user doesn't specify
	min_question_count: int = 2  # Minimum questions allowed
	max_question_count: int = 10  # Maximum questions allowed
	default_experience_years: int = 2  # Default experience for fallback profile
	default_domain: str = "Software Engineer"  # Default domain for fallback

	@field_validator("answer_temperature")
	@classmethod
	def clamp_temperature(cls, v: float) -> float:
		return max(0.0, min(1.0, v))

	@field_validator("cors_allow_origins", mode="before")
	@classmethod
	def parse_cors_origins(cls, v):
		# Allow environment variable override
		if isinstance(v, str):
			return [origin.strip() for origin in v.split(",")]
		return v

settings = Settings()
