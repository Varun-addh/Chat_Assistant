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
	groq_model: str = "moonshotai/kimi-k2-instruct-0905"
	answer_temperature: float = 0.4
	groq_top_p: float | None = None
	groq_max_tokens: int | None = None  # Override automatic token limit calculation
	groq_max_tokens_simple: int = 300  # For simple questions
	groq_max_tokens_code: int = 800  # For code questions
	groq_max_tokens_complex: int = 1200  # For complex topics
	groq_reasoning_effort: str | None = None  # e.g., "medium"
	groq_stream: bool = False

	# Google Gemini
	gemini_api_key: str | None = None
	# Update default to Gemini 3 preview model. Can be overridden via env var (GEMINI_MODEL).
	gemini_model: str = "models/gemini-flash-latest"

	# SerpAPI (for faster, more reliable web search)
	serpapi_api_key: str | None = None

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
