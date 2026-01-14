from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import List
import secrets
import os
import re
from dotenv import load_dotenv


def _env_files() -> list[str]:
	"""Return env file(s) to load.

	Supports comma-separated ENV_FILE to allow layered configs, e.g.:
	- ENV_FILE=.env,.env.local
	- ENV_FILE=.env,.env.docker

	If ENV_FILE is not set, defaults to .env.
	"""
	raw = (os.getenv("ENV_FILE", ".env") or ".env").strip()
	files = [p.strip() for p in raw.split(",") if p.strip()]
	return files or [".env"]


# Ensure env files are loaded eagerly (highest priority is OS env vars, then env files).
for _p in _env_files():
	load_dotenv(dotenv_path=_p, override=False)


class Settings(BaseSettings):
	model_config = SettingsConfigDict(
		env_file=_env_files(),
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
	# Environment name (development/staging/production). Used to toggle certain defaults.
	app_env: str = "development"
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
	# Demo-mode model override for Groq.
	# If set, demo requests will be restricted to this model (no fallback) to keep
	# cost/behavior predictable. If not set, demo uses the normal groq_model.
	groq_demo_model: str | None = None
	# Stratax-controlled platform key used ONLY for Demo Mode (no-auth + no user key).
	# This key must never be exposed to clients. Keep strict demo quotas to cap cost.
	stratax_demo_api_key: str | None = None
	# Optional pool (comma-separated or JSON list). If present, the backend will
	# automatically rotate and cooldown exhausted demo keys.
	stratax_demo_api_keys: List[str] = []
	# If false, demo users will NOT use STRATAX demo keys. This is recommended for
	# local development so your UI testing uses your developer/server keys instead.
	# In production, set this to true to enable real public demo traffic.
	enable_demo_key_pool: bool = False
	# Safety: even if ENABLE_DEMO_KEY_POOL=true, do NOT allow consuming demo keys
	# in development/local unless you explicitly opt in.
	allow_demo_key_pool_in_dev: bool = False
	# Hard global demo budget guard (daily). When exceeded, demo returns 503.
	demo_global_daily_request_limit: int = 0  # 0 = disabled
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
	# Event logging / telemetry (data moat foundation)
	# - analytics_hmac_key: if set, we use HMAC-SHA256 for stable, non-reversible IDs.
	# - analytics_store_raw_text: if true, we may store question/answer text in event payloads.
	#   Keep this false by default to reduce privacy risk.
	analytics_hmac_key: str | None = None
	analytics_store_raw_text: bool = False
	analytics_text_preview_len: int = 120
	enable_event_logging: bool = True

	# Feature Flags
	enable_hybrid_search: bool = True
	enable_reranking: bool = True
	enable_code_execution: bool = True
	enable_query_expansion: bool = True
	enable_streaming: bool = True

	# Qdrant (Vector DB)
	# If qdrant_url is set, Interview Intelligence will connect to a Qdrant server.
	# If not set, it will fall back to local-path Qdrant (single-process only).
	qdrant_url: str | None = None  # env: QDRANT_URL
	qdrant_api_key: str | None = None  # env: QDRANT_API_KEY
	qdrant_collection_name: str = "interview_questions"  # env: QDRANT_COLLECTION_NAME

	# Startup / performance flags
	# Useful for tests, CI, and lightweight deployments.
	# When enabled, we skip expensive model/service initialization in app startup.
	fast_startup: bool = False
	# Force-disable Interview Intelligence startup even if optional deps are installed.
	disable_interview_intelligence: bool = False

	# Redis (rate limiting + cross-worker caches)
	# If REDIS_URL is set, rate limiting and selected caches can be backed by Redis,
	# enabling safe multi-instance scaling.
	redis_url: str | None = None  # env: REDIS_URL
	redis_key_prefix: str = "stratax"  # env: REDIS_KEY_PREFIX
	# If true (default), Redis outages should not take the API down; we fail-open.
	redis_fail_open: bool = True  # env: REDIS_FAIL_OPEN
	# TTL for shared caches (seconds)
	redis_cache_ttl_seconds: int = 3600  # env: REDIS_CACHE_TTL_SECONDS
	
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

	@field_validator("stratax_demo_api_keys", mode="before")
	@classmethod
	def parse_demo_key_pool(cls, v):
		def _expand_env_ref(s: str) -> str | None:
			# Support referencing other env vars inside JSON lists, e.g. ["${GROQ_API_KEY}"]
			# This avoids duplicating secrets in .env while keeping pydantic-settings JSON decoding happy.
			s = (s or "").strip()
			if not s:
				return None
			m = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", s)
			if m:
				val = os.environ.get(m.group(1), "").strip()
				return val or None
			m2 = re.fullmatch(r"\$([A-Za-z_][A-Za-z0-9_]*)", s)
			if m2:
				val = os.environ.get(m2.group(1), "").strip()
				return val or None
			return s

		# Accept JSON list (pydantic may parse automatically), or comma-separated string.
		if v is None:
			return []
		if isinstance(v, list):
			out: list[str] = []
			for x in v:
				expanded = _expand_env_ref(str(x))
				if expanded:
					out.append(expanded)
			return out
		if isinstance(v, str):
			raw = v.strip()
			if not raw:
				return []
			parts = [p.strip() for p in raw.split(",") if p.strip()]
			out: list[str] = []
			for p in parts:
				expanded = _expand_env_ref(p)
				if expanded:
					out.append(expanded)
			return out
		return v

	def get_effective_provider(self, feature: str = "default") -> str:
		# NOTE: demo key pool enablement is handled separately by is_demo_key_pool_enabled().
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

	def is_demo_key_pool_enabled(self) -> bool:
		"""Effective demo key pool enablement.

		Goal:
		- Prevent developers from accidentally burning demo keys while testing locally.
		- Allow public demo traffic to use demo key pool in production/staging.

		Rules:
		- If ENABLE_DEMO_KEY_POOL is false -> disabled.
		- In production/staging: enabled when ENABLE_DEMO_KEY_POOL is true.
		- In dev/test/local: enabled only if both ENABLE_DEMO_KEY_POOL=true AND
		  ALLOW_DEMO_KEY_POOL_IN_DEV=true.
		"""
		if not bool(self.enable_demo_key_pool):
			return False
		env = (self.app_env or "development").strip().lower()
		if env in {"production", "prod", "staging", "stage"}:
			return True
		# development / local safety gate
		return bool(self.allow_demo_key_pool_in_dev)

settings = Settings()


def get_settings() -> Settings:
	"""Backwards-compatible settings accessor.

	Several modules/tests expect a callable to retrieve the singleton Settings.
	"""
	return settings
