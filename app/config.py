from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices, field_validator, model_validator
from typing import List
import secrets
import logging
import os
import re
from dotenv import load_dotenv


def _env_files() -> list[str]:
	"""Return env file(s) to load.

	Supports comma-separated ENV_FILE to allow layered configs, e.g.:
	- ENV_FILE=.env,.env.local
	- ENV_FILE=.env,.env.docker

	If ENV_FILE is not set, default to `.env` and automatically layer in
	`.env.local` when present. This keeps machine-local overrides working for
	local development without requiring shell setup.
	"""
	raw = os.getenv("ENV_FILE")
	if raw is None or not raw.strip():
		files = [".env"]
		if os.path.exists(".env.local"):
			files.append(".env.local")
		return files

	files = [p.strip() for p in raw.split(",") if p.strip()]
	return files or [".env"]


# Ensure env files are loaded eagerly.
# Priority order (highest -> lowest): OS env vars > later env files > earlier env files.
# We achieve this by loading files in reverse order with override=False.
for _p in reversed(_env_files()):
	load_dotenv(dotenv_path=_p, override=False)


class Settings(BaseSettings):
	model_config = SettingsConfigDict(
		env_file=_env_files(),
		env_file_encoding="utf-8",
		extra="ignore",
	)

	# Product Identity (used in assistant responses)
	app_name: str = "Stratax AI"
	# IMPORTANT: Do not hard-code ownership/employment/affiliation claims.
	# If you want public attribution, set these explicitly via environment variables.
	app_developer_name: str = ""
	app_developer_attribution: str = (
		"Stratax AI is an independently developed platform. For official information about its development or ownership, "
		"please refer to Stratax AI’s documentation or website."
	)

	# Server
	host: str = "0.0.0.0"
	port: int = 8000
	# Environment name (development/staging/production). Used to toggle certain defaults.
	app_env: str = "development"
	cors_allow_origins: List[str] = ["*"]
	max_request_body_bytes: int = 25 * 1024 * 1024  # 25 MB default ceiling for generic HTTP bodies
	max_practice_media_upload_bytes: int = 100 * 1024 * 1024  # Keep media uploads aligned with Practice Mode route caps
	# Public base URL used to build absolute redirect URIs (OAuth callbacks, etc.)
	backend_base_url: str = "http://localhost:8000"
	# Frontend URL for redirecting back after OAuth
	frontend_url: str = "http://localhost:8080"

	# Auth
	api_key: str | None = None  # simple bearer key if provided
	cookie_secret: str | None = Field(default=None, validation_alias=AliasChoices("COOKIE_SECRET", "STRATAX_COOKIE_SECRET"))
	jwt_secret_key: str | None = Field(default=None, validation_alias=AliasChoices("JWT_SECRET_KEY", "STRATAX_JWT_SECRET_KEY"))  # JWT secret for authentication
	# Encryption key for user-stored provider secrets (Fernet key). Recommended for production.
	# Generate with:
	#   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
	secrets_encryption_key: str | None = Field(
		default=None,
		validation_alias=AliasChoices("STRATAX_SECRETS_ENCRYPTION_KEY", "SECRETS_ENCRYPTION_KEY"),
	)
	
	# Sentry error tracking (production only)
	sentry_dsn: str | None = Field(default=None, validation_alias="SENTRY_DSN")
	app_version: str = Field(default="1.0.0", validation_alias="APP_VERSION")
	
	# Database
	database_url: str = Field(
		default="sqlite:///./data/stratax.db",
		validation_alias=AliasChoices("DATABASE_URL", "STRATAX_DATABASE_URL")
	)

	# Practice Learning (privacy-safe aggregates only)
	# When enabled, the app stores small per-session aggregate metrics and optional
	# self-reported outcomes (no raw audio, no raw transcripts).
	enable_practice_learning: bool = Field(
		default=False,
		validation_alias=AliasChoices("ENABLE_PRACTICE_LEARNING", "STRATAX_ENABLE_PRACTICE_LEARNING"),
	)

	# Premium: deterministic adaptive pressure rules.
	# When enabled, the practice follow-up generator may adjust difficulty/tone deterministically.
	enable_adaptive_pressure: bool = Field(
		default=False,
		validation_alias=AliasChoices("ENABLE_ADAPTIVE_PRESSURE", "STRATAX_ENABLE_ADAPTIVE_PRESSURE"),
	)

	# Email (verification + password reset)
	# If email_enabled is false, the API will still generate tokens and (in dev)
	# can return URLs for manual testing, but it won't attempt SMTP.
	email_enabled: bool | None = None  # env: EMAIL_ENABLED; auto-enables in production when SMTP is configured
	email_from: str | None = Field(default=None, validation_alias=AliasChoices("EMAIL_FROM", "SMTP_FROM"))
	smtp_host: str | None = Field(default=None, validation_alias=AliasChoices("SMTP_HOST", "EMAIL_SMTP_HOST"))
	smtp_port: int = Field(default=587, validation_alias=AliasChoices("SMTP_PORT", "EMAIL_SMTP_PORT"))
	smtp_username: str | None = Field(default=None, validation_alias=AliasChoices("SMTP_USERNAME", "EMAIL_SMTP_USERNAME"))
	smtp_password: str | None = Field(default=None, validation_alias=AliasChoices("SMTP_PASSWORD", "EMAIL_SMTP_PASSWORD"))
	smtp_use_tls: bool = Field(default=True, validation_alias=AliasChoices("SMTP_USE_TLS", "EMAIL_SMTP_USE_TLS"))
	smtp_use_ssl: bool = Field(default=False, validation_alias=AliasChoices("SMTP_USE_SSL", "EMAIL_SMTP_USE_SSL"))

	# Token expirations
	email_verification_token_ttl_minutes: int = 60 * 24  # env: EMAIL_VERIFICATION_TOKEN_TTL_MINUTES (default 24h)
	password_reset_token_ttl_minutes: int = 30  # env: PASSWORD_RESET_TOKEN_TTL_MINUTES

	@model_validator(mode="after")
	def _require_persistent_secrets_in_production(self):
		# In dev/test we allow auto-generated secrets for convenience.
		# In production, secrets MUST be persistent or all sessions/tokens break on restart.
		env = (self.app_env or "").strip().lower()
		if env == "production":
			if not (os.getenv("JWT_SECRET_KEY") or os.getenv("STRATAX_JWT_SECRET_KEY")):
				raise ValueError("Missing JWT_SECRET_KEY in production")
			if not (os.getenv("COOKIE_SECRET") or os.getenv("STRATAX_COOKIE_SECRET")):
				raise ValueError("Missing COOKIE_SECRET in production")
			# Prevent accidental SQLite usage in production.
			if (self.database_url or "").strip().lower().startswith("sqlite"):
				raise ValueError("SQLite is not allowed in production. Set DATABASE_URL to a Postgres URL.")
			# If email is enabled, require SMTP config.
			if bool(self.email_enabled):
				if not (self.email_from and self.email_from.strip()):
					raise ValueError("EMAIL_FROM is required when EMAIL_ENABLED=true in production")
				if not (self.smtp_host and self.smtp_host.strip()):
					raise ValueError("SMTP_HOST is required when EMAIL_ENABLED=true in production")
				if self.smtp_username and not self.smtp_password:
					raise ValueError("SMTP_PASSWORD is required when SMTP_USERNAME is set")
		return self

	@field_validator("jwt_secret_key", mode="before")
	@classmethod
	def _default_jwt_secret_key(cls, v):
		# ONLY auto-generate in dev/test environments
		env = os.getenv("APP_ENV", "development").strip().lower()
		if v is None or (isinstance(v, str) and not v.strip()):
			if env in ("production", "staging"):
				# Force explicit configuration in production
				raise ValueError(
					"JWT_SECRET_KEY must be explicitly set in production/staging. "
					"Generate with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
				)
			# Dev/test convenience - warn about ephemeral secret
			logging.warning("Using auto-generated JWT secret (dev only). Tokens will be invalid after restart.")
			return secrets.token_urlsafe(32)
		return v

	@field_validator("cookie_secret", mode="before")
	@classmethod
	def _default_cookie_secret(cls, v):
		env = os.getenv("APP_ENV", "development").strip().lower()
		if v is None or (isinstance(v, str) and not v.strip()):
			if env in ("production", "staging"):
				raise ValueError(
					"COOKIE_SECRET must be explicitly set in production/staging. "
					"Generate with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
				)
			logging.warning("Using auto-generated cookie secret (dev only). Sessions will be invalid after restart.")
			return secrets.token_urlsafe(32)
		return v

	@field_validator("email_enabled", mode="before")
	@classmethod
	def _default_email_enabled(cls, v):
		if isinstance(v, bool):
			return v
		if isinstance(v, str) and v.strip():
			value = v.strip().lower()
			if value in {"1", "true", "yes", "on"}:
				return True
			if value in {"0", "false", "no", "off"}:
				return False

		env = os.getenv("APP_ENV", "development").strip().lower()
		has_smtp_host = bool((os.getenv("SMTP_HOST") or os.getenv("EMAIL_SMTP_HOST") or "").strip())
		has_from_email = bool((os.getenv("EMAIL_FROM") or os.getenv("SMTP_FROM") or "").strip())
		if env == "production" and has_smtp_host and has_from_email:
			return True
		return False

	# LLM Provider Selection
	llm_provider: str = "gemini"  # default global provider; Interview Intelligence overrides to Groq internally

	# Cohere
	cohere_api_key: str | None = None

	# Judge0
	judge0_api_key: str | None = Field(
		default=None,
		validation_alias=AliasChoices("JUDGE0_API_KEY", "RAPIDAPI_KEY"),
	)
	# RapidAPI host header for Judge0. Only override this if you know you need it.
	# (Do NOT set this to some other RapidAPI host like google-translate.)
	judge0_rapidapi_host: str = Field(default="judge0-ce.p.rapidapi.com", validation_alias="JUDGE0_RAPIDAPI_HOST")

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
	# Dynamic budgeting: how many tokens equals one budget unit. The DynamicBudgetEngine
	# converts intent/depth/length units -> tokens using this multiplier. Tweak to
	# reflect model/token pricing and desired verbosity per unit.
	token_per_budget_unit: int = 400

	# LLM intent-detection keywords (deterministic heuristics)
	# Centralizing these avoids duplicated hardcoded lists across services.
	llm_comparison_keywords: List[str] = ["compare", "versus", "vs ", "difference between", "differences between"]
	llm_greeting_exact: List[str] = [
		"hi", "hello", "hey", "yo", "hiya", "heya", "good morning", "good afternoon", "good evening", "gm", "gn",
		"thank you", "thanks", "thx", "ty", "bye", "goodbye", "see you", "see ya", "cya", "take care",
	]
	llm_greeting_prefixes: List[str] = [
		"hi ", "hello ", "hey ", "thank you", "thanks", "thx", "ty ", "good morning", "good afternoon", "good evening",
		"bye", "goodbye", "see you", "see ya",
	]
	llm_greeting_name_intro_prefixes: List[str] = ["my name is ", "call me ", "i am ", "im ", "i'm "]
	llm_greeting_assist_phrases: List[str] = ["how can i help you", "how can i assist you", "how may i help you", "how may i assist you"]
	llm_off_topic_keywords: List[str] = [
		"weather", "news", "politics", "sports", "entertainment", "personal advice", "relationship", "health", "medical",
		"cooking", "travel", "shopping", "finance", "investment", "current events", "celebrity", "movie", "music", "book",
		"game", "gaming", "social media", "dating", "family",
	]
	llm_off_topic_patterns: List[str] = [
		"what's happening", "what's new", "how's your day", "tell me about yourself personally", "what do you think about",
		"do you know about", "have you heard about", "what's your opinion",
	]
	llm_ambiguous_patterns: List[str] = ["how do you", "what about", "tell me about", "explain", "what is", "how does", "why", "when", "where"]
	llm_ambiguous_technical_terms: List[str] = [
		"algorithm", "data structure", "database", "api", "framework", "language", "coding", "programming", "system", "design",
		"interview", "technical", "behavioral", "experience",
	]
	llm_personal_indicators: List[str] = [
		"yourself", "myself", "about you", "about me", "your background", "my background", "your experience", "my experience",
		"your skills", "my skills", "your strengths", "my strengths", "your weaknesses", "my weaknesses", "your projects", "my projects",
		"your career", "my career", "your goals", "my goals", "hire you", "interested in", "motivates you", "motivates me",
		"introduce", "tell me about", "describe yourself",
	]
	llm_personal_references: List[str] = [
		"you are", "you have", "you did", "you worked", "you developed", "you created", "you built", "you designed", "you implemented",
	]
	llm_strategy_indicators: List[str] = [
		"optimize", "improve", "reduce", "increase", "solve", "handle", "implement", "approach", "strategy", "method", "technique",
		"performance", "efficiency", "scalability", "reliability",
	]
	llm_strategy_question_patterns: List[str] = ["how", "what", "which", "describe", "explain"]
	llm_strategy_personal_indicators: List[str] = [
		"tell me about yourself", "your experience", "your background", "your skills", "your strengths", "your weaknesses", "your projects",
		"why should we hire you", "what motivates you", "introduce yourself",
	]
	llm_database_schema_keywords: List[str] = [
		"database schema", "er diagram", "entity relationship", "database design", "show the database", "database structure", "table design",
		"schema design", "relational model", "database model", "data model",
	]
	llm_ui_design_keywords: List[str] = [
		"front page", "user interface", "ui design", "mobile app interface", "frontend design", "ui/ux", "user experience", "wireframe",
		"mockup", "prototype", "visual design", "layout design", "design the front", "design the interface", "design the page",
	]
	llm_algorithm_keywords: List[str] = [
		"algorithm", "data structure", "sorting", "searching", "recommendation algorithm", "build a recommendation", "implement authentication",
		"authentication algorithm", "search algorithm", "matching algorithm", "optimization algorithm", "prime", "binary", "tree", "graph",
		"stack", "queue", "linked list", "recursion", "logic", "solve", "math", "complexity", "big o",
	]

	# Interview Intelligence heuristics
	interview_intelligence_company_keywords: List[str] = ["google", "amazon", "facebook", "meta", "microsoft", "apple", "netflix"]
	interview_intelligence_requires_code_keywords: List[str] = [
		"coding", "algorithm", "implement", "code", "write a function", "program", "solution",
	]

	# Document analyzer heuristics
	document_jd_keywords: List[str] = [
		"responsibilities", "requirements", "qualifications", "we are looking for", "job description", "position", "reports to",
		"salary range", "benefits",
	]
	document_resume_keywords: List[str] = ["experience", "education", "skills", "projects", "certifications", "achievements", "resume", "cv"]
	document_cover_letter_keywords: List[str] = ["dear", "sincerely", "i am writing", "i am interested", "cover letter", "application for"]

	# Local TTS voice selection heuristics
	# NOTE: Avoid overly-broad keywords like "man" which can false-match language names
	# (e.g., "Manipuri") and cause surprising voice selection.
	tts_male_voice_keywords: List[str] = ["david", "mark", "alex", "james", "george", "male"]
	tts_female_voice_keywords: List[str] = ["zira", "hazel", "female", "woman", "samantha", "victoria"]

	# Architecture view selection heuristics
	architecture_async_keywords: List[str] = ["event", "queue", "async", "worker", "background", "kafka", "rabbitmq", "sqs"]
	architecture_data_keywords: List[str] = ["database", "cache", "redis", "postgres", "mongodb", "elasticsearch", "search"]
	architecture_security_keywords: List[str] = ["auth", "security", "oauth", "jwt", "rbac", "permission"]

	# Evaluation fallback: detect code structure from raw text
	evaluation_code_indicators: List[str] = [
		"def ", "class ", "import ", "from ", "include ", "using ", "public ", "private ", "static ", "void ", "int ", "float ",
		"func ", "let ", "const ", "var ", "function ", "async ", "await", "print(", "console.log", "return ", "yield ",
		"if (", "for (", "while (", "{", "}", ":", "(", ")",
	]

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
	# If true, Practice Mode runs via a LangGraph workflow (optional dependency).
	# Defaults to false to keep deployments simple.
	enable_practice_mode_langgraph: bool = False

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
	# Optional safety valve for burst traffic: caps concurrent CPU/IO heavy operations
	# (embeddings + vector search offloads) per worker.
	# None = unlimited (current behavior).
	embedding_concurrency_limit: int | None = None  # env: EMBEDDING_CONCURRENCY_LIMIT

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
	practice_tts_engine: str = "edge_tts"  # options: edge_tts (neural, recommended), pyttsx3 (offline), gtts (online)
	# TTS tuning
	practice_tts_timeout_seconds: float = 30.0
	practice_tts_disable_gtts_fallback: bool = False
	# Optional hard overrides for voice selection (pyttsx3/SAPI). If set, these win over keyword matching.
	practice_tts_voice_id: str | None = None
	practice_tts_voice_name_contains: str | None = None
	practice_tts_model: str = "default"  # Not used for pyttsx3/gtts
	# Edge-TTS neural voice. Options: en-US-GuyNeural, en-US-ChristopherNeural, en-US-EricNeural, en-US-SteffanNeural
	practice_edge_tts_voice: str = "en-US-GuyNeural"
	practice_stt_model_size: str = "base"  # options: tiny, base, small, medium, large
	practice_stt_device: str = "cpu"  # options: cpu, cuda
	practice_stt_max_transcription_time_seconds: float = 3.0  # env: PRACTICE_STT_MAX_TRANSCRIPTION_TIME_SECONDS
	practice_stt_target_rtf: float = 0.35  # env: PRACTICE_STT_TARGET_RTF
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
