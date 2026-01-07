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

	# Product Identity (used in assistant responses)
	app_name: str = "Stratax AI"
	app_developer_name: str = "Varun Bikkumalla"
	app_developer_attribution: str = (
		"Stratax AI is powered by advanced language models and engineered by Varun Bikkumalla (sole developer)."
	)

	# Server
	host: str = "0.0.0.0"
	port: int = 8000
	cors_allow_origins: List[str] = ["*"]
	# Public base URL used to build absolute redirect URIs (OAuth callbacks, etc.)
	backend_base_url: str = "http://localhost:8000"
	# Frontend URL for redirecting back after OAuth
	frontend_url: str = "http://localhost:8080"

	# Auth
	api_key: str | None = None  # simple bearer key if provided
	cookie_secret: str = secrets.token_urlsafe(32)
	jwt_secret_key: str = secrets.token_urlsafe(32)  # JWT secret for authentication

	# LLM Provider Selection
	llm_provider: str = "gemini"  # default global provider; Interview Intelligence overrides to Groq internally

	# Cohere
	cohere_api_key: str | None = None

	# Judge0
	judge0_api_key: str | None = None

	# Groq
	groq_api_key: str | None = None
	groq_model: str = "llama-3.3-70b-versatile"
	groq_fallback_models: List[str] = [
		"openai/gpt-oss-120b",
		"openai/gpt-oss-20b",
		"qwen/qwen3-32b",
		"meta-llama/llama-4-scout-17b-16e-instruct",
		"llama-3.1-8b-instant"
	]
	answer_temperature: float = 1
	groq_top_p: float | None = 1
	groq_max_tokens: int | None = None  # Set to None to enable smart dynamic calculation (saves tokens!)
	groq_max_tokens_simple: int = 500  # For simple questions (increased from 300)
	groq_max_tokens_code: int = 2000  # For code questions (increased from 800)
	groq_max_tokens_complex: int = 3000  # For complex topics (increased from 2500 for resume reviews)
	groq_reasoning_effort: str | None = None  # e.g., "medium"
	groq_stream: bool = True

	# Google Gemini
	gemini_api_key: str | None = None
	# Update default to Gemini 3 preview model. Can be overridden via env var (GEMINI_MODEL).
	gemini_model: str = "models/gemini-3-flash-preview"

	# Google OAuth (Login with Google)
	google_client_id: str | None = None
	google_client_secret: str | None = None

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

	# Startup / performance flags
	# Useful for tests, CI, and lightweight deployments.
	# When enabled, we skip expensive model/service initialization in app startup.
	fast_startup: bool = False
	# Force-disable Interview Intelligence startup even if optional deps are installed.
	disable_interview_intelligence: bool = False
	
	# API Key Policy
	require_user_api_key: bool = False  # If True, users MUST provide their own API key (server keys won't be used as fallback)

	# Practice Mode Configuration
	practice_mode_enabled: bool = True
	practice_tts_engine: str = "pyttsx3"  # options: pyttsx3 (offline), gtts (online) - pyttsx3 for better male voice & speed control
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

	# Architecture Detection / Diagram Complexity (deterministic heuristics)
	# NOTE: These defaults can be overridden via environment variables using JSON
	# list syntax if desired (pydantic-settings supports JSON parsing for List).
	architecture_detection_explicit_keywords: List[str] = [
		"system design",
		"architecture",
		"high level design",
		"hld",
	]
	architecture_detection_design_verbs: List[str] = [
		"design a",
		"design an",
		"design the",
		"build a",
		"build an",
		"create a system",
		"develop a platform",
	]
	architecture_detection_system_concepts_scale: List[str] = [
		"scalability",
		"scale",
		"distributed",
		"microservice",
	]
	architecture_detection_system_concepts_data: List[str] = [
		"load balanc",
		"database design",
		"data model",
	]
	architecture_detection_system_concepts_infra: List[str] = [
		"deployment",
		"infrastructure",
		"cloud",
	]
	architecture_detection_code_problem_keywords: List[str] = [
		"implement a function",
		"write a function",
		"write code",
		"implement a class",
		"algorithm for",
		"code to",
		"function that",
	]

	# Signals used to estimate requirements complexity for adaptive diagram sizing.
	# These are intentionally simple substring matches (fast, deterministic).
	architecture_complexity_signals: List[str] = [
		# scale/infra
		"scale",
		"scalable",
		"high throughput",
		"low latency",
		"sub-second",
		"global",
		"multi-region",
		"multi region",
		"high availability",
		"fault tolerance",
		"disaster recovery",
		"geo",
		"cdn",
		"load balancer",
		# correctness/consistency
		"consistency",
		"strong consistency",
		"exactly once",
		"idempot",
		"transaction",
		"acid",
		"double",
		"locking",
		# domain complexity
		"payment",
		"billing",
		"refund",
		"inventory",
		"pricing",
		"recommend",
		"search",
		"analytics",
		"auditing",
		# security/compliance
		"oauth",
		"jwt",
		"rbac",
		"pii",
		"gdpr",
		"soc2",
		"hipaa",
		"encryption",
		# async/eventing
		"queue",
		"kafka",
		"pubsub",
		"event",
		"worker",
		"stream",
		"sse",
		"websocket",
	]

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

	def get_effective_provider(self, feature: str = "default") -> str:
		"""
		Determine which LLM provider to use based on available API keys.
		
		Logic:
		- If only Groq API key: use Groq for all features
		- If only Gemini API key: use Gemini for all features  
		- If both API keys: use Gemini for AI Copilot Chat (feature="copilot"), 
		  Groq for all other features (Interview Intelligence, Mock Interview, etc.)
		
		Args:
			feature: "copilot" for AI Copilot Chat Assistant, "default" for other features
			
		Returns:
			"groq" or "gemini"
		"""
		has_groq = bool(self.groq_api_key and self.groq_api_key.strip())
		has_gemini = bool(self.gemini_api_key and self.gemini_api_key.strip())
		
		# If only one provider configured, use it for everything
		if has_groq and not has_gemini:
			return "groq"
		if has_gemini and not has_groq:
			return "gemini"
		
		# Both providers configured: Gemini for Copilot, Groq for everything else
		if has_groq and has_gemini:
			if feature == "copilot":
				return "gemini"
			return "groq"
		
		# No providers configured: fallback to configured default
		return self.llm_provider

settings = Settings()


def get_settings() -> Settings:
	"""Backwards-compatible settings accessor.

	Several modules/tests expect a callable to retrieve the singleton Settings.
	"""
	return settings
