from __future__ import annotations

from typing import AsyncIterator, Optional, List, Dict, Any, Tuple
from groq import Groq
try:
	from app.services.chat.gemini_adapter import genai
except Exception:
    genai = None

from app.config import settings
import asyncio
import logging
import json
import re
import threading

from app.services.chat.demo_key_pool import demo_key_pool

from app.prompts.builder import PromptFlags, build_default_system_prompt
from app.prompts.policies import (
	DEPTH_BUDGET,
	OUTPUT_GUARDS,
	OUTPUT_HYGIENE,
	PROMPT_INJECTION_RESISTANCE,
	RESPONSE_CONTRACT,
	RESPONSE_TEMPLATE,
)
from app.prompts.response_plan import ResponsePlan
from app.services.chat.mirror_ontology import MirrorOntologyGenerator
from app.schemas import MirrorReport
from app.services.chat.dynamic_budget import dynamic_budget_engine
from app.services.llm import (
	ambiguous_query_overrides,
	algorithm_overrides,
	comparison_overrides,
	context_fallback_overrides,
	database_schema_overrides,
	greeting_overrides,
	groq_models_to_try,
	identity_overrides,
	identity_response_text,
	is_identity_question,
	is_system_design_question,
	off_topic_overrides,
	persona_overrides,
	system_design_overrides,
	technical_strategy_overrides,
	ui_design_overrides,
)

from app.services.llm import response_postprocess

logger = logging.getLogger(__name__)


class LLMAuthenticationError(Exception):
	"""Raised when an upstream LLM provider rejects the configured credentials."""


def _is_quota_or_rate_limit_error(exc: Exception) -> bool:
	msg = str(exc).lower()
	return any(
		x in msg
		for x in [
			"429",
			"rate_limit",
			"rate limit",
			"quota",
			"resource_exhausted",
			"exceeded",
			"too many requests",
		]
	)


def _is_authentication_error(exc: Exception) -> bool:
	msg = str(exc).lower()
	return any(
		x in msg
		for x in [
			"401",
			"unauthorized",
			"invalid_api_key",
			"invalid api key",
			"authentication_error",
			"authentication failed",
		]
	)


# Lock to protect genai.configure() calls (global state in the legacy library)
_genai_configure_lock = threading.Lock()


def _configure_genai(api_key: str):
	"""Thread-safe wrapper around genai.configure()."""
	with _genai_configure_lock:
		genai.configure(api_key=api_key)
		return genai


# NOTE: Legacy mega-prompt removed.
# Prompt construction now uses policy composition in app/prompts/* via build_default_system_prompt.


class LLMService:
	def __init__(self) -> None:
		self._client: Groq | None = None
		self._settings = settings  # Will be overridden by factory
		self._mirror_ontology = MirrorOntologyGenerator()

	def _get_app_identity(self) -> tuple[str, str, str]:
		"""Returns (app_name, developer_name, attribution)."""
		return (
			self._settings.app_name,
			self._settings.app_developer_name,
			self._settings.app_developer_attribution,
		)

	def _groq_models_to_try(
		self,
		*,
		groq_model_override: Optional[str] = None,
		restrict_to_override: bool = False,
		limit: Optional[int] = None,
	) -> list[str]:
		"""Return an ordered, de-duplicated list of Groq models to try."""
		return groq_models_to_try(
			self._settings,
			groq_model_override=groq_model_override,
			restrict_to_override=restrict_to_override,
			limit=limit,
		)

	def _identity_response_text(self, question: str) -> str:
		return identity_response_text(self._settings, question)

	def _is_identity_question(self, question: str) -> bool:
		return is_identity_question(self._settings, question)

	def _identity_overrides(self) -> str:
		return identity_overrides(self._settings)

	async def generate_text(
		self, 
		prompt: str, 
		system_prompt: Optional[str] = None,
		api_key: Optional[str] = None,
		json_mode: bool = False,
		temperature: float = 0.3,
		max_tokens: int = 2000,
		model: Optional[str] = None,
		raise_on_auth_error: bool = False,
	) -> str:
		"""
		🚀 UNIVERSAL GENERATOR - One method to rule them all.
		Intelligently routes requests to Groq or Gemini based on API key prefix.
		Supports JSON mode and System prompts across all providers.
		"""
		# Deterministic identity answers (avoid LLM hallucinated attribution)
		# IMPORTANT: Only apply this to *direct user questions*.
		# Many internal calls (Mirror mode prompts, ontology generation, rewrite passes)
		# contain identity keywords in system policies or are multi-line composite prompts.
		# Short-circuiting those would return non-JSON and break downstream parsing.
		p = (prompt or "").strip()
		is_direct_user_query = (
			(not json_mode)
			and (system_prompt is None or not str(system_prompt).strip())
			and ("\n" not in p)
			and (len(p) <= 200)
			and (not p.lower().startswith("interview question:"))
		)
		if is_direct_user_query and self._is_identity_question(p):
			logger.info("🪪 [IDENTITY] generate_text short-circuit: %s", p[:200])
			return self._identity_response_text(p)

		def _call(local_key: Optional[str]):
			client, provider = self._ensure_client(local_key)
			if client is None:
				raise Exception("LLM client not available. Please configure an API key.")

			try:
				if provider == "groq":
					messages = []
					if system_prompt:
						messages.append({"role": "system", "content": system_prompt})
					messages.append({"role": "user", "content": prompt})
					
					extra_args = {}
					if json_mode:
						extra_args["response_format"] = {"type": "json_object"}
					
					target_model = model or self._settings.groq_model
					resp = client.chat.completions.create(
						model=target_model,
						messages=messages,
						temperature=temperature,
						max_tokens=max_tokens,
						**extra_args
					)
					return resp.choices[0].message.content
				
				elif provider == "gemini":
					target_model = model or self._settings.gemini_model
					if target_model.startswith("models/"):
						target_model = target_model.replace("models/", "")
					
					# Use system_instruction for proper role separation (avoids prompt injection weakness)
					model_kwargs = {}
					if system_prompt:
						model_kwargs["system_instruction"] = system_prompt
					gmodel = client.GenerativeModel(target_model, **model_kwargs)
					
					config = {
						"temperature": temperature,
						"max_output_tokens": max_tokens
					}
					if json_mode:
						config["response_mime_type"] = "application/json"
					
					# Permissive on technical content, but keep basic safety for harmful content
					safety_settings = [
						{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
						{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
						{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
						{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
					]
					
					resp = gmodel.generate_content(
						prompt,
						generation_config=config,
						safety_settings=safety_settings
					)
					if hasattr(resp, "text"):
						return resp.text
					elif hasattr(resp, "candidates") and resp.candidates:
						return resp.candidates[0].content.parts[0].text
					return ""
				
				else:
					raise ValueError(f"Unsupported provider: {provider}")
					
			except Exception as e:
				logger.error(f"LLM Error ({provider}): {e}")
				raise

		import anyio

		# Only consider demo key pool when it's explicitly enabled in settings.
		if settings.is_demo_key_pool_enabled():
			pool_keys = set(demo_key_pool.keys())
		else:
			pool_keys = set()
		attempt_key = api_key
		for attempt in range(2):
			try:
				result = await anyio.to_thread.run_sync(_call, attempt_key)
				return (result or "").strip()
			except Exception as e:
				if _is_authentication_error(e):
					logger.error(f"Generate text authentication failed: {e}")
					if raise_on_auth_error:
						raise LLMAuthenticationError(str(e)) from e
				if attempt == 0 and attempt_key and attempt_key in pool_keys and _is_quota_or_rate_limit_error(e):
					demo_key_pool.mark_exhausted(attempt_key, reason=str(e)[:200])
					attempt_key = demo_key_pool.get_key()  # rotate
					if attempt_key:
						logger.warning("[DEMO_KEY_POOL] Retrying generate_text with a different demo key")
						continue
				logger.error(f"Generate text failed: {e}")
				return ""



	def _needs_comparison(self, question: str) -> bool:
		q = (question or "").lower()
		keywords = getattr(self._settings, "llm_comparison_keywords", None) or []
		return any(k in q for k in keywords)

	def _is_greeting(self, question: str) -> bool:
		q = (question or "").strip().lower()
		if not q:
			return False
		# Normalize common punctuation
		for ch in ["!", ".", ","]:
			q = q.replace(ch, "")
		# Collapse repeated whitespace
		q = re.sub(r"\s+", " ", q).strip()
		# Single/two-word salutations and courtesies
		greetings = set(getattr(self._settings, "llm_greeting_exact", None) or [])
		# Quick exact match
		if q in greetings:
			return True
		# Startswith match for polite variants
		prefixes = list(getattr(self._settings, "llm_greeting_prefixes", None) or [])
		if any(q.startswith(p) for p in prefixes):
			return True

		# Name introductions / role-confusion style openers (keep conservative)
		name_intro_phrases = list(getattr(self._settings, "llm_greeting_name_intro_prefixes", None) or [])
		assist_phrases = list(getattr(self._settings, "llm_greeting_assist_phrases", None) or [])
		# Treat short name-intros and helper-offers as greetings
		if any(p in q for p in assist_phrases):
			return True
		if any(q.startswith(p) for p in name_intro_phrases) and len(q.split()) <= 12:
			return True
		return False

	def _is_off_topic(self, question: str) -> bool:
		"""Detect if question is off-topic for interview preparation"""
		q = (question or "").lower()
		if not q.strip():
			return False
		
		# Off-topic indicators
		off_topic_keywords = list(getattr(self._settings, "llm_off_topic_keywords", None) or [])
		
		# Check for off-topic keywords
		if any(keyword in q for keyword in off_topic_keywords):
			return True
		
		# Check for non-interview question patterns
		off_topic_patterns = list(getattr(self._settings, "llm_off_topic_patterns", None) or [])
		
		return any(pattern in q for pattern in off_topic_patterns)

	def _is_ambiguous(self, question: str) -> bool:
		"""Detect if question is ambiguous and needs clarification.

		Uses word count instead of character length to avoid flagging valid
		short technical questions like 'TCP vs UDP' (3 words, 10 chars).
		"""
		q = (question or "").strip()
		words = q.split()
		# Single-word or empty queries are genuinely ambiguous
		if len(words) <= 1:
			return True
		
		# Ambiguous patterns from config
		ambiguous_patterns = list(getattr(self._settings, "llm_ambiguous_patterns", None) or [])
		
		if any(pattern in q.lower() for pattern in ambiguous_patterns):
			# Only flag as ambiguous if ALSO very short AND no technical terms
			if len(words) < 4:
				technical_terms = list(getattr(self._settings, "llm_ambiguous_technical_terms", None) or [])
				if not any(term in q.lower() for term in technical_terms):
					return True
		
		return False

	def _has_sufficient_context(self, question: str, previous_qna: Optional[List[Dict[str, str]]]) -> bool:
		"""Check if this question is a standalone query that does NOT need context.

		Returns True when the question stands on its own (sufficient context in itself).
		Returns False when the question references prior conversation (needs history).
		When previous_qna is empty, the question is always standalone → True.
		"""
		if not previous_qna or len(previous_qna) == 0:
			# No history = standalone question = sufficient context
			return True
		
		q = (question or "").lower()
		words = set(q.split())
		
		# Pronouns that reference prior conversation (word-boundary via set lookup)
		context_pronouns = {"this", "that", "it", "these", "those", "them"}
		if words & context_pronouns:
			return False  # needs context
		
		# Explicit back-references
		context_references = ["previous", "earlier", "above", "before", "last"]
		if any(ref in q for ref in context_references):
			return False  # needs context
		
		# Continuation patterns
		follow_up_patterns = {"also", "additionally", "furthermore"}
		if words & follow_up_patterns:
			return False  # needs context
		
		return True  # standalone question

	def _greeting_overrides(self) -> str:
		return greeting_overrides()

	def _off_topic_overrides(self) -> str:
		return off_topic_overrides()

	def _ambiguous_query_overrides(self) -> str:
		return ambiguous_query_overrides()

	def _context_fallback_overrides(self) -> str:
		return context_fallback_overrides()

	def _comparison_overrides(self, question: str) -> str:
		return comparison_overrides(question)

	def _needs_first_person(self, question: str) -> bool:
		q = (question or "").lower()
		
		# If it's a technical strategy question, don't use first person
		if self._is_technical_strategy_question(question):
			return False
		
		# Look for personal/behavioral question indicators
		personal_indicators = list(getattr(self._settings, "llm_personal_indicators", None) or [])
		
		# Look for direct personal references
		personal_references = list(getattr(self._settings, "llm_personal_references", None) or [])
		
		# Check for personal indicators or references
		has_personal_indicator = any(indicator in q for indicator in personal_indicators)
		has_personal_reference = any(reference in q for reference in personal_references)
		
		return has_personal_indicator or has_personal_reference

	def _is_technical_strategy_question(self, question: str) -> bool:
		"""Check if this is a technical strategy question that should provide general approaches"""
		q = (question or "").lower()
		
		# Look for strategy/approach indicators
		strategy_indicators = list(getattr(self._settings, "llm_strategy_indicators", None) or [])
		
		# Look for question patterns that suggest strategy/approach
		question_patterns = list(getattr(self._settings, "llm_strategy_question_patterns", None) or [])
		
		# Check if it's asking for a method/approach rather than personal experience
		has_strategy_indicator = any(indicator in q for indicator in strategy_indicators)
		has_question_pattern = any(pattern in q for pattern in question_patterns)
		
		# Look for specific personal experience indicators that would override strategy mode
		personal_indicators = list(getattr(self._settings, "llm_strategy_personal_indicators", None) or [])
		
		has_personal_indicator = any(indicator in q for indicator in personal_indicators)
		
		# If it has strategy indicators and question patterns, but NOT personal indicators, it's a strategy question
		return has_strategy_indicator and has_question_pattern and not has_personal_indicator

	def _is_system_design_question(self, question: str) -> bool:
		"""Detect explicit System Design / Architecture questions."""
		return is_system_design_question(question)

	def _is_database_schema_question(self, question: str) -> bool:
		"""Detect database schema / ER diagram questions"""
		q = (question or "").lower()
		keywords = list(getattr(self._settings, "llm_database_schema_keywords", None) or [])
		return any(k in q for k in keywords)

	def _is_ui_design_question(self, question: str) -> bool:
		"""Detect UI/UX design questions"""
		q = (question or "").lower()
		keywords = list(getattr(self._settings, "llm_ui_design_keywords", None) or [])
		defaults = [
			"ui",
			"ux",
			"user interface",
			"user experience",
			"layout",
			"wireframe",
			"design a screen",
			"design a page",
			"settings page",
			"dashboard",
			"form",
			"navigation",
		]
		# Always include defaults (misconfig-safe), but keep behavior stable if user added their own list.
		kw = [k for k in (keywords + defaults) if str(k).strip()]
		return any(k in q for k in kw)

	def _is_algorithm_question(self, question: str) -> bool:
		"""Detect algorithm and data structure questions"""
		q = (question or "").lower()
		keywords = list(getattr(self._settings, "llm_algorithm_keywords", None) or [])
		return any(k in q for k in keywords)

	def _database_schema_overrides(self) -> str:
		"""Overrides for database schema questions"""
		return database_schema_overrides()

	def _ui_design_overrides(self) -> str:
		"""Overrides for UI design questions"""
		return ui_design_overrides()

	def _algorithm_overrides(self) -> str:
		"""Overrides for algorithm questions"""
		return algorithm_overrides()

	def _system_design_overrides(self) -> str:
		"""Enforce the System Design response structure requested by the user."""
		return system_design_overrides()

	def _technical_strategy_overrides(self) -> str:
		return technical_strategy_overrides()

	def _persona_overrides(self) -> str:
		return persona_overrides()

	def _ensure_client_by_provider(self, provider_name: str, api_key: Optional[str] = None) -> tuple[any, str]:
		"""Returns (client, provider) for a specific provider by name."""
		name = provider_name.lower()
		
		# If user provided key, try to use it
		if api_key:
			if name == "groq": return Groq(api_key=api_key), "groq"
			if name == "gemini":
				if genai: 
					return _configure_genai(api_key), "gemini"
		
		# Fallback to server keys
		if name == "groq" and self._settings.groq_api_key:
			return Groq(api_key=self._settings.groq_api_key), "groq"
		if name == "gemini" and self._settings.gemini_api_key:
			if genai:
				return _configure_genai(self._settings.gemini_api_key), "gemini"
		
		return None, name

	def _ensure_client(self, api_key: Optional[str] = None) -> tuple[any, str]:
		"""Returns (client, provider)"""
		provider = (self._settings.llm_provider or "groq").lower()
		
		# Clean and validate API key
		if api_key:
			api_key = api_key.strip()
			if api_key.lower() in ("null", "undefined", "none", ""):
				api_key = None

		# Demo key pooling: only active when demo key pool is enabled. If the
		# provided api_key matches a configured demo key, rotate to an available
		# demo key to handle cooldowns after quota/rate-limit failures.
		if api_key and settings.is_demo_key_pool_enabled():
			pool_keys = set(demo_key_pool.keys())
			if api_key in pool_keys and len(pool_keys) > 1:
				api_key = demo_key_pool.get_key(preferred=api_key) or api_key
		
		# If API key is provided, detect provider from key prefix
		if api_key:
			if "gsk_" in api_key:
				return Groq(api_key=api_key), "groq"
			elif "AIza" in api_key or "ATza" in api_key:
				if genai is None:
					logger.warning("Gemini key detected but google-generativeai not installed")
					return None, "gemini"
				return _configure_genai(api_key), "gemini"
			
			# Fallback to globally configured provider if prefix not recognized
			logger.debug("API key prefix not recognized (not gsk_/AIza/ATza), using configured provider: %s", provider)
			if provider == "groq":
				return Groq(api_key=api_key), "groq"
			elif provider == "gemini":
				if genai is None:
					return None, "gemini"
				return _configure_genai(api_key), "gemini"
			else:
				# If provider is unknown but we have a key, default to Groq for safety
				return Groq(api_key=api_key), "groq"
		
		# No user-provided API key - check if we should use server's key as fallback
		if self._settings.require_user_api_key:
			return None, provider

		# Permissive mode: Use server's API key as fallback
		if provider == "groq":
			api_key = self._settings.groq_api_key
			if not api_key:
				self._client = None
				return None, "groq"
			if self._client is None or not isinstance(self._client, Groq):
				self._client = Groq(api_key=api_key)
			return self._client, "groq"
		elif provider == "gemini":
			if genai is None:
				return None, "gemini"
			api_key = self._settings.gemini_api_key
			if not api_key:
				return None, "gemini"
			return _configure_genai(api_key), "gemini"
		else:
			return None, provider

	@property
	def enabled(self) -> bool:
		provider = (self._settings.llm_provider or "groq").lower()
		if provider == "groq":
			return bool(self._settings.groq_api_key)
		if provider == "gemini":
			return bool(self._settings.gemini_api_key)
		return False

	async def generate_algorithm_frames(self, problem: str, code: str, language: str) -> List[Dict[str, str]]:
		"""Ask the model to emit STRICT JSON frames describing step-by-step Mermaid diagrams.

		Return value: List[ {"mermaid": str, "caption": str} ] with 3–12 frames.
		If the provider is not configured, return an empty list.
		"""
		client, provider = self._ensure_client()
		if client is None:
			return []

		prompt = (
			"You are producing step-by-step VISUALIZATION FRAMES for how the algorithm executes.\n"
			"Output STRICT JSON ONLY — no prose. The JSON schema is: {\n"
			"  \"frames\": [ { \"mermaid\": string, \"caption\": string } ]\n"
			"}.\n\n"
			"Rules for frames (avoid extra legend boxes):\n"
			"- 3 to 12 frames, each a small delta from previous.\n"
			"- Use Mermaid flowchart TD or LR. Prefer LR for array-like steps.\n"
			"- Show array contents and highlight current key/comparisons using class or styles.\n"
			"- Use arrows to indicate movement or comparison.\n"
			"- Use classDef to color: key (fill:#ffecb3,stroke:#ff9800), compare (fill:#e1f5fe,stroke:#0288d1), fixed (fill:#e8f5e9,stroke:#2e7d32).\n"
			"- Do NOT create separate legend nodes like 'Data Layer' or 'Client Layer'. If grouping is needed, use 'subgraph' with titles only; no floating boxes.\n"
			"- Keep the canvas minimal: only nodes that participate in the current step.\n"
			"- Keep labels short.\n"
			"- Do NOT include markdown fences.\n"
			"- Escape newlines and quotes for valid JSON.\n\n"
			"Context provided:\n"
			f"Problem: {problem or 'N/A'}\n"
			f"Language: {language}\n"
			"Code snippet follows. Derive the algorithm (e.g., insertion sort) and produce frames accordingly.\n"
			"Focus on: array state per outer loop, key element insertion, shifts, comparisons.\n"
		)

		user_payload = (
			f"```{language}\n{code}\n```\n"
		)

		import anyio, json as _json
		def _call():
			if provider == "groq":
				messages: List[Dict[str, str]] = [
					{"role": "system", "content": prompt},
					{"role": "user", "content": user_payload},
				]
				resp = client.chat.completions.create(
					model=self._settings.groq_model,
					messages=messages,
					temperature=0.2,
					max_tokens=self._settings.groq_max_tokens or 8000,
				)
				return resp.choices[0].message.content
			elif provider == "gemini":
				gmodel = client.GenerativeModel(self._settings.gemini_model)
				full_prompt = prompt + "\n\nUser:\n" + user_payload
				resp = gmodel.generate_content(full_prompt)
				return getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "")
			else:
				return ""

		try:
			raw = await anyio.to_thread.run_sync(_call)
			text = (raw or "").strip()
			# Extract JSON robustly
			start = text.find("{")
			end = text.rfind("}")
			if start != -1 and end != -1 and end > start:
				obj = _json.loads(text[start:end+1])
				frames = obj.get("frames") or []
				clean: List[Dict[str, str]] = []
				for item in frames:
					merm = (item.get("mermaid") or "").strip()
					cap = (item.get("caption") or "").strip()
					if merm:
						clean.append({"mermaid": merm, "caption": cap})
				return clean
		except Exception:
			return []


	def _estimate_response_complexity(self, question: str) -> int:
		"""Estimate response complexity and suggest token limit"""
		question_lower = question.lower()
		
		# Simple questions - shorter responses
		simple_indicators = ['what is', 'define', 'explain briefly', 'simple', 'basic']
		if any(indicator in question_lower for indicator in simple_indicators):
			return self._settings.groq_max_tokens_simple
		
		# Code questions - medium responses
		code_indicators = ['code', 'implement', 'write', 'function', 'class', 'algorithm']
		if any(indicator in question_lower for indicator in code_indicators):
			return self._settings.groq_max_tokens_code
		
		# Complex topics - longer responses
		complex_indicators = [
			'architecture', 'design', 'system', 'compare', 'difference',
			'advantages', 'disadvantages', 'best practices', 
			'resume', 'cv', 'profile', 'review', 'analyze'
		]
		if any(indicator in question_lower for indicator in complex_indicators):
			return self._settings.groq_max_tokens_complex
		
		# Default medium response (average of simple and code)
		return (self._settings.groq_max_tokens_simple + self._settings.groq_max_tokens_code) // 2

	def _infer_depth_level(self, question: str) -> str:
		"""Infer a simple depth knob from the user's request.

		Returns one of: 'quick', 'standard', 'deep'.
		"""
		q = (question or "").lower()
		# Explicit deep requests
		if any(k in q for k in ["in depth", "in-depth", "deep dive", "deep-dive", "comprehensive", "thorough", "detailed", "teach me", "explain deeply", "explain in detail", "explain in detail:", "in detail"]):
			return "deep"
		# Explicit brevity requests
		if any(k in q for k in ["brief", "quick", "tl;dr", "tldr", "summary only", "in short", "one-liner", "one liner"]):
			return "quick"
		return "standard"

	def _flags_from_plan(self, plan: ResponsePlan, question: str) -> PromptFlags:
		"""Convert ResponsePlan to PromptFlags (single source of truth for intent)."""
		return PromptFlags(
			is_system_design=(plan.intent == "system_design"),
			is_database_schema=(plan.intent == "database_schema"),
			is_ui_design=(plan.intent == "ui_design"),
			is_algorithm=(plan.intent == "coding"),
			needs_first_person=self._needs_first_person(question),
			is_technical_strategy=(plan.intent == "technical_strategy"),
			is_mirror_mode=(plan.intent == "mirror"),
			intent=plan.intent or "",
			depth=plan.depth,
		)

	def _normalize_depth(self, depth: Optional[str], question: str) -> str:
		"""Normalize an explicit depth override (UI/API) or fall back to inference."""
		if depth is not None:
			d = (depth or "").strip().lower()
			if d:
				aliases: dict[str, str] = {
					"quick": "quick",
					"brief": "quick",
					"concise": "quick",
					"short": "quick",
					"standard": "standard",
					"default": "standard",
					"normal": "standard",
					"deep": "deep",
					"detailed": "deep",
					"deep-dive": "deep",
					"deepdive": "deep",
					"thorough": "deep",
					"comprehensive": "deep",
				}
				mapped = aliases.get(d)
				if mapped is not None:
					return mapped
		# Fall back to inference from the question text.
		return self._infer_depth_level(question)

	def _infer_format_hint(self, question: str, style_mode: Optional[str], layout: Optional[str]) -> str:
		q = (question or "").lower()
		# UI/automation: explicitly requested JSON
		if "json" in q and any(k in q for k in ["only json", "return json", "output json", "respond in json"]):
			return "json"
		if (layout or "").lower() in {"qa", "faq", "checklist"}:
			return "text"
		if self._is_algorithm_question(question) or any(k in q for k in ["code", "implement", "write a function", "leetcode"]):
			return "code"
		# If the user explicitly chooses a style preset that implies more prose, keep it textual.
		if (style_mode or "").lower() in {"narrative", "executive", "mentor"}:
			return "text"
		return "text"

	def _build_response_plan(
		self,
		question: str,
		*,
		depth: Optional[str] = None,
		style_mode: Optional[str] = None,
		layout: Optional[str] = None,
		mode: Optional[str] = None,
	) -> ResponsePlan:
		"""Create a deterministic routing plan (intent/depth/format)."""
		mode_norm = (mode or "").strip().lower()
		if mode_norm == "mirror":
			intent = "mirror"
		elif self._is_greeting(question):
			intent = "greeting"
		elif self._is_off_topic(question):
			intent = "off_topic"
		elif self._is_system_design_question(question):
			intent = "system_design"
		elif self._is_database_schema_question(question):
			intent = "database_schema"
		elif self._is_ui_design_question(question):
			intent = "ui_design"
		elif self._is_algorithm_question(question) or any(k in (question or "").lower() for k in ["code", "implement", "write", "function", "class", "leetcode"]):
			intent = "coding"
		elif self._is_technical_strategy_question(question):
			intent = "technical_strategy"
		else:
			intent = "general"

		resolved_depth = self._normalize_depth(depth, question)
		# Mirror mode always needs a JSON payload internally.
		if intent == "mirror":
			fmt = "json"
		else:
			fmt = self._infer_format_hint(question, style_mode, layout)
		return ResponsePlan(intent=intent, depth=resolved_depth, format=fmt)

	def _get_optimal_token_limit(
		self,
		question: str,
		base_limit: int,
		*,
		depth: Optional[str] = None,
		mode: Optional[str] = None,
		user_tier: str = "standard",
	) -> int:
		"""Get optimal token limit based on question intent and depth budget.

		Important: some intents (system design/coding/etc.) need a higher minimum
		budget to avoid mid-answer truncation. We keep the DynamicBudgetEngine as
		the primary allocator, but apply an intent-aware minimum floor.
		"""
		# Use the DynamicBudgetEngine to compute an intent/depth/length-aware budget.
		# We infer intent/format via the existing planner to pass a meaningful intent.
		plan = self._build_response_plan(question, depth=depth, mode=mode)
		intent = plan.intent or "general"
		resolved_depth = plan.depth or (depth or "standard")
		# Ask the engine to compute tokens; pass base_limit as a hard ceiling candidate.
		target = dynamic_budget_engine.compute_budget_tokens(
			question=question,
			intent=intent,
			depth=resolved_depth,
			user_tier=user_tier,
			base_limit=base_limit,
		)

		# Apply a higher minimum for long-form intents to prevent cutoffs.
		long_form_intents = {
			"system_design",
			"coding",
			"database_schema",
			"ui_design",
			"technical_strategy",
			"mirror",
		}
		if intent in long_form_intents:
			# Base minimum from existing heuristic buckets.
			min_budget = int(self._estimate_response_complexity(question) or 0)
			depth_mult = {"quick": 0.75, "standard": 1.0, "deep": 1.25}.get(resolved_depth, 1.0)
			min_budget = int(min_budget * depth_mult)
			# Keep a sane lower bound even if heuristic fails.
			min_budget = max(900, min_budget)
			target = max(target, min_budget)

		# Respect ceilings (explicit base_limit, else configured complex cap).
		ceiling = None
		if base_limit:
			ceiling = int(base_limit)
		elif getattr(self._settings, "groq_max_tokens_complex", None):
			ceiling = int(getattr(self._settings, "groq_max_tokens_complex"))
		if ceiling is not None:
			target = min(int(target), int(ceiling))

		return int(target)

	def _extract_json_object(self, text: str) -> Optional[dict[str, Any]]:
		"""Best-effort extraction of a JSON object from model output.

		This is intentionally defensive: some models wrap JSON in prose, code fences,
		or include other text. We try direct parse first, then fall back to a
		brace-matching scan that respects JSON strings/escapes.
		"""
		raw = (text or "").strip()
		if not raw:
			return None
		# Strip common fences
		raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
		raw = re.sub(r"\s*```$", "", raw).strip()

		try:
			obj = json.loads(raw)
			if isinstance(obj, dict):
				return obj
		except Exception:
			pass

		# Fallback: brace-matching scan for the first valid JSON object.
		# This avoids regex pitfalls with nested braces and braces inside strings.
		start = raw.find("{")
		while start != -1:
			depth = 0
			in_string = False
			escaped = False
			for i in range(start, len(raw)):
				ch = raw[i]
				if in_string:
					if escaped:
						escaped = False
						continue
					if ch == "\\":
						escaped = True
						continue
					if ch == '"':
						in_string = False
					continue

				# Not in a string
				if ch == '"':
					in_string = True
					continue
				if ch == "{":
					depth += 1
					continue
				if ch == "}":
					depth -= 1
					if depth < 0:
						break
					if depth == 0:
						candidate = raw[start : i + 1]
						try:
							obj = json.loads(candidate)
							if isinstance(obj, dict):
								return obj
						except Exception:
							# Keep scanning from the next '{'
							break
						# If it parsed but isn't a dict, keep scanning.
						break

			# Next candidate
			start = raw.find("{", start + 1)
		return None

	def _calibrate_mirror_confidence(
		self,
		*,
		llm_confidence: float,
		ontology: Any,
		question: str,
		user_answer: str,
		strengths: list[str],
		gaps: list[str],
		schema_drift: bool,
	) -> tuple[float, dict[str, Any]]:
		"""Calibrate Mirror confidence using deterministic signals.

		We keep the LLM's confidence as an input, but down/up-weight it based on:
		- schema drift
		- answer specificity/length
		- coverage of inferred primitives
		- ratio of strengths vs gaps
		"""
		q = (question or "").strip().lower()
		a = (user_answer or "").strip().lower()
		c = self._normalize_confidence(llm_confidence)
		meta: dict[str, Any] = {"llm_confidence": c}

		# Schema drift: treat as unreliable even if the model claims high confidence.
		if schema_drift:
			c = min(c, 0.35)
			meta["schema_drift_penalty"] = True

		# Answer specificity heuristics
		answer_len = len(a)
		meta["answer_len"] = answer_len
		is_definition_q = bool(re.search(r"\b(what\s+is|define|meaning\s+of)\b", q))
		meta["is_definition_q"] = is_definition_q

		# Token-based vagueness proxy (very cheap)
		toks = re.findall(r"[a-z]{3,}", a)
		uniq = len(set(toks))
		meta["unique_tokens"] = uniq

		if answer_len < 60:
			# For definition questions, short can still be meaningful. Penalize more only if extremely vague.
			if is_definition_q:
				if uniq <= 4:
					c = c * 0.75
					meta["short_definition_penalty"] = True
				else:
					c = c * 0.9
					meta["short_definition_light_penalty"] = True
			else:
				# Non-definition: short answers are more likely underspecified.
				c = c * 0.7
				meta["short_penalty"] = True

			# Extremely short answers are rarely trustworthy outside of trivial definitions.
			if answer_len < 35 and (not is_definition_q):
				c = min(c, 0.35)
				meta["too_short_cap"] = True
		elif answer_len < 140:
			c = c * 0.9
			meta["medium_length_penalty"] = True
		else:
			meta["length_ok"] = True

		# Primitive coverage (word-boundary matching; avoids false positives from substrings)
		primitives = list(getattr(ontology, "primitives", ()) or ())
		primitives = [str(p).strip().lower() for p in primitives if str(p).strip()]
		matched = 0
		for p in primitives[:8]:
			if p and self._primitive_present(p, a):
				matched += 1
		coverage = (matched / max(1, min(5, len(primitives)))) if primitives else 0.0
		meta["primitive_coverage"] = round(coverage, 3)

		if primitives:
			if coverage < 0.2:
				c = max(0.0, c - 0.15)
				meta["low_coverage_penalty"] = True
			elif coverage > 0.6 and answer_len >= 140:
				# If the user covered most primitives, allow a small bump when the model is pessimistic.
				if c < 0.55:
					c = min(1.0, c + 0.1)
					meta["coverage_bump"] = True

		# Strengths vs gaps ratio sanity
		gap_n = len(gaps or [])
		strength_n = len(strengths or [])
		meta["gaps_count"] = gap_n
		meta["strengths_count"] = strength_n
		if gap_n >= 4 and strength_n == 0:
			c = max(0.0, c - 0.1)
			meta["gap_heavy_penalty"] = True

		# If the question looks like a deep design question, be more conservative.
		if any(k in q for k in ("design", "architecture", "trade-off", "scal", "throughput")) and answer_len < 180:
			c = min(c, 0.39 if answer_len < 80 else 0.5)
			meta["design_conservatism"] = True

		c = self._normalize_confidence(c)
		meta["calibrated_confidence"] = c
		return c, meta

	def _normalize_string_list(self, value: Any, *, limit: int) -> list[str]:
		if value is None:
			items: list[Any] = []
		elif isinstance(value, list):
			items = value
		else:
			items = [value]
		out: list[str] = []
		for item in items:
			if item is None:
				continue
			s = str(item).strip()
			if not s:
				continue
			out.append(s)
			if len(out) >= limit:
				break
		return out

	def _normalize_confidence(self, value: Any) -> float:
		try:
			c = float(value)
		except Exception:
			c = 0.5
		if c < 0.0:
			return 0.0
		if c > 1.0:
			return 1.0
		return c

	def _looks_like_meta_advice(self, line: str) -> bool:
		l = (line or "").strip().lower()
		if not l:
			return False
		# Heuristic: coaching/instruction phrasing rather than something the candidate can say.
		bad_starts = (
			"provide ",
			"include ",
			"mention ",
			"emphasize ",
			"make sure",
			"ensure ",
			"try to ",
			"you should",
			"focus on",
			"talk about",
		)
		if any(l.startswith(s) for s in bad_starts):
			return True
		# Also catch common meta words.
		meta_tokens = ("comprehensive definition", "more detail", "be more", "explicitly mention")
		return any(t in l for t in meta_tokens)

	def _looks_harsh(self, line: str) -> bool:
		l = (line or "").strip().lower()
		if not l:
			return False
		# Detect demotivating/absolute phrasing in red flags.
		harsh_words = (
			"confuses",
			"wrong",
			"overly simplistic",
			"fundamentally flawed",
			"completely missing",
			"shows no understanding",
			"fails to",
		)
		return any(h in l for h in harsh_words)

	async def _rewrite_upgrade_lines_if_needed(
		self,
		*,
		question: str,
		user_answer: str,
		topic: str,
		upgrade_lines: list[str],
		api_key: Optional[str],
	) -> list[str]:
		if not upgrade_lines:
			return []
		if not any(self._looks_like_meta_advice(x) for x in upgrade_lines):
			return upgrade_lines

		system_prompt = (
			"Rewrite coaching-style upgrade advice into interview-ready sentences. "
			"Return ONLY JSON: {\"upgrade_lines\": [string]}. "
			"No extra keys. No prose."
		)
		user_prompt = (
			"Interview question:\n"
			+ (question or "").strip()
			+ "\n\nUser draft answer:\n"
			+ (user_answer or "").strip()
			+ "\n\nTopic (inferred): "
			+ (topic or "General")
			+ "\n\nThese are NOT acceptable because they are meta-advice:\n- "
			+ "\n- ".join([x.strip() for x in upgrade_lines if x.strip()])
			+ "\n\nRules:\n"
			"- Output 1–2 lines the candidate can say verbatim.\n"
			"- Each line should be a complete sentence.\n"
			"- For definition questions, one line should be a crisp one-sentence definition.\n"
			"- Do NOT use instruction verbs like 'mention', 'provide', 'emphasize'.\n"
		)

		raw = await self.generate_text(
			user_prompt,
			system_prompt=system_prompt,
			api_key=api_key,
			json_mode=True,
			temperature=0.2,
			max_tokens=250,
		)
		obj = self._extract_json_object(raw)
		if not obj:
			return []
		return self._normalize_string_list(obj.get("upgrade_lines"), limit=2)

	async def _soften_red_flags_if_needed(
		self,
		*,
		question: str,
		user_answer: str,
		topic: str,
		red_flags: list[str],
		api_key: Optional[str],
	) -> list[str]:
		if not red_flags:
			return []
		if not any(self._looks_harsh(x) for x in red_flags):
			return red_flags

		system_prompt = (
			"Rewrite harsh red-flag phrasing into supportive risk framing. "
			"Return ONLY JSON: {\"red_flags\": [string]}. "
			"No extra keys. No prose." 
			"Use allowed phrases such as 'May sound junior', 'Narrow framing', 'Missing senior signals' and avoid words like 'wrong', 'confuses', 'fundamentally flawed'."
		)
		user_prompt = (
			"Interview question:\n"
			+ (question or "").strip()
			+ "\n\nUser draft answer:\n"
			+ (user_answer or "").strip()
			+ "\n\nTopic (inferred): "
			+ (topic or "General")
			+ "\n\nThese red flags sound too harsh:\n- "
			+ "\n- ".join([x.strip() for x in red_flags if x.strip()])
			+ "\n\nRules:\n"
			"- Rewrite each as a supportive risk observation.\n"
			"- Use the allowed phrasing examples from the system prompt (e.g., 'May sound junior', 'Narrow framing', 'Missing senior signals').\n"
			"- Avoid absolutes like 'confuses', 'wrong', 'overly simplistic', 'fundamentally flawed'.\n"
			"- Keep it factual but not demotivating.\n"
		)

		raw = await self.generate_text(
			user_prompt,
			system_prompt=system_prompt,
			api_key=api_key,
			json_mode=True,
			temperature=0.2,
			max_tokens=200,
		)
		obj = self._extract_json_object(raw)
		if not obj:
			return []
		return self._normalize_string_list(obj.get("red_flags"), limit=3)

	def _format_mirror_markdown(self, report: dict[str, Any]) -> str:
		topic = (report.get("topic") or "General").strip()
		message = (report.get("message") or "").strip()
		strengths = report.get("strengths") or []
		gaps = report.get("gaps") or []
		red_flags = report.get("red_flags") or []
		followups = report.get("likely_followups") or []
		upgrade = report.get("upgrade_lines") or []
		confidence = report.get("confidence")

		lines: list[str] = []
		lines.append(f"## Interview Mirror ({topic})")
		if message:
			lines.append(f"\n{message}")

		if strengths:
			lines.append("\n**Strengths**")
			lines.extend([f"- {s}" for s in strengths])
		if gaps:
			lines.append("\n**Gaps to Close**")
			lines.extend([f"- {s}" for s in gaps])
		if red_flags:
			lines.append("\n**Red Flags**")
			lines.extend([f"- {s}" for s in red_flags])
		if followups:
			lines.append("\n**Likely Follow-ups**")
			lines.extend([f"- {s}" for s in followups])
		if upgrade:
			lines.append("\n**Upgrade Lines (say this instead)**")
			lines.extend([f"- {s}" for s in upgrade])

		try:
			c = float(confidence)
			lines.append(f"\nConfidence: {int(round(c * 100))}%")
		except Exception:
			pass

		return "\n".join(lines).strip() + "\n"

	def _primitive_present(self, primitive: str, user_answer: str) -> bool:
		"""Check if a primitive concept is present using word-boundary matching."""
		try:
			# Escape regex special chars in the primitive, then use word boundaries
			pattern = r'\b' + re.escape(primitive) + r'\b'
			return bool(re.search(pattern, user_answer))
		except re.error:
			# Fallback: space-padded containment (safer than bare `in`)
			return f" {primitive} " in f" {user_answer} "

	def _enforce_mirror_gap_and_followup_policy(self, report: dict[str, Any], ontology: Any, user_answer: str) -> dict[str, Any]:
		"""Merge must-have primitive gaps with LLM-provided gaps and sharpen follow-ups.

		- Uses word-boundary matching (not bare substring) to detect primitives.
		- Merges must-have gaps with LLM gaps (does not overwrite).
		- Sharpens follow-ups to prefer concept-probing questions.
		"""
		if not report or not ontology:
			return report
		ua = (user_answer or "").lower()
		primitives = [str(p).strip().lower() for p in list(getattr(ontology, "primitives", ()) or []) if str(p).strip()]
		must_gaps: list[str] = []
		for p in primitives:
			if p and not self._primitive_present(p, ua):
				must_gaps.append(f"Missing core concept: {p}")

		# Merge: must-have gaps first, then LLM-provided gaps (deduped), up to limit
		llm_gaps = self._normalize_string_list(report.get("gaps"), limit=5)
		merged: list[str] = list(must_gaps)
		seen_lower = {g.lower() for g in merged}
		for g in llm_gaps:
			if g.lower() not in seen_lower and len(merged) < 5:
				merged.append(g)
				seen_lower.add(g.lower())
		report["gaps"] = self._normalize_string_list(merged, limit=5)

		# Sharpen follow-ups: prefer questions starting with key probing verbs
		followups = list(report.get("likely_followups") or [])
		preferred: list[str] = []
		for f in followups:
			fs = (f or "").strip()
			if not fs:
				continue
			if re.match(r'^(explain|how|what|compare|describe|why)\b', fs.strip().lower()):
				preferred.append(fs)

		if preferred:
			report["likely_followups"] = self._normalize_string_list(preferred, limit=3)
		else:
			report["likely_followups"] = self._normalize_string_list(followups, limit=3)

		return report

	def _apply_low_confidence_guard(self, report: dict[str, Any]) -> dict[str, Any]:
		"""Trust guardrails: prevent false authority when confidence is low."""
		confidence = self._normalize_confidence(report.get("confidence"))
		report["confidence"] = confidence
		meta = report.get("_meta") if isinstance(report.get("_meta"), dict) else {}
		# Brevity acknowledgment: estimate word count from answer_len in calibration meta
		ua_words = 0
		if isinstance(meta.get("confidence_details"), dict) and isinstance(meta["confidence_details"].get("answer_len"), int):
			ua_words = max(0, meta["confidence_details"]["answer_len"] // 5)
		elif isinstance(meta.get("answer_len"), int):
			ua_words = max(0, meta.get("answer_len") // 5)

		brevity_note = ""
		if ua_words and ua_words < 15:
			brevity_note = "Answer is very brief; treating as partial signal. "

		# Full low-confidence guard
		if confidence < 0.40:
			meta["low_confidence_guard"] = True
			report["upgrade_lines"] = []
			# Clamp gaps to 2 and prefer must-have primitives (enforced elsewhere)
			report["gaps"] = self._normalize_string_list(report.get("gaps"), limit=2)
			# Strong, explicit message per request (include canonical phrase for tests)
			report["message"] = f"{brevity_note}Low confidence: Answer is too brief to assess reliably; please expand."
			report["_meta"] = meta
			return report

		# Partial guard: tentative assessment for 0.40–0.55
		if 0.40 <= confidence < 0.55:
			meta["partial_confidence_guard"] = True
			# Keep upgrade_lines (allow), but clamp gaps to 2
			report["gaps"] = self._normalize_string_list(report.get("gaps"), limit=2)
			# Add brief disclaimer about tentative assessment
			disclaimer = "Assessment tentative (limited confidence); gaps are provisional."
			report["message"] = ((brevity_note or "") + ("Low confidence (tentative): " + str(report.get("message") or "").strip() + " " + disclaimer)).strip()
			report["_meta"] = meta
			return report

		# Otherwise, confident enough — return unchanged (but ensure meta present)
		report["_meta"] = meta
		return report

	def _mirror_expected_keys(self) -> set[str]:
		return {
			"topic",
			"message",
			"strengths",
			"gaps",
			"red_flags",
			"likely_followups",
			"upgrade_lines",
			"confidence",
		}

	async def generate_mirror_report_structured(
		self,
		*,
		question: str,
		user_answer: str,
		depth: Optional[str] = None,
		api_key: Optional[str] = None,
	) -> Tuple[str, bool, dict[str, Any]]:
		"""Generate an Interview Mirror report.

		Contract:
		- Calls the LLM in strict JSON mode.
		- Validates and clamps the output.
		- Returns a UI-friendly markdown summary.
		"""
		q = (question or "").strip()
		ua = (user_answer or "").strip()
		if not q:
			return ("", False, {})
		if not ua:
			return ("", False, {})

		plan = self._build_response_plan(q, depth=depth, mode="mirror")
		flags = self._flags_from_plan(plan, q)
		app_name, developer, attribution = self._get_app_identity()
		system_prompt = build_default_system_prompt(
			app_name=app_name,
			developer_name=developer,
			attribution=attribution,
			flags=flags,
		)

		ontology = await self._mirror_ontology.get(
			question=q,
			generate_text=self.generate_text,
			api_key=api_key,
		)
		guidance = (
			"Analysis guidance (do not output this text; use it to judge gaps):\n"
			f"- Inferred topic: {ontology.topic}\n"
			"- Expected primitives: " + "; ".join(ontology.primitives) + "\n"
			"- Senior signals: " + "; ".join(ontology.senior_signals) + "\n"
			"- Common red flags: " + "; ".join(ontology.red_flags) + "\n"
			"- Likely follow-ups: " + "; ".join(ontology.likely_followups) + "\n"
			"- IMPORTANT: If a concept is clearly IMPLIED by what the user said, do NOT mark it as missing.\n"
		)

		user_prompt = (
			"Interview question:\n"
			+ q
			+ "\n\nUser draft answer:\n"
			+ ua
			+ "\n\n"
			+ guidance
			+ "\n\nReturn the JSON object only."
		)

		max_tokens = self._get_optimal_token_limit(q, self._settings.groq_max_tokens, depth=depth, mode="mirror")
		raw = await self.generate_text(
			user_prompt,
			system_prompt=system_prompt,
			api_key=api_key,
			json_mode=True,
			temperature=0.2,
			max_tokens=max_tokens,
		)

		obj = self._extract_json_object(raw)
		if obj is None:
			# Safe fallback: never crash the request due to model formatting drift.
			fallback = {
				"topic": ontology.topic or "General",
				"message": "Low confidence: the Mirror report couldn't be parsed. Ask a clarifying follow-up and retry.",
				"strengths": [],
				"gaps": ["Could not parse a structured Mirror report. Please retry."],
				"red_flags": [],
				"likely_followups": [],
				"upgrade_lines": [],
				"confidence": 0.2,
				"_meta": {
					"parse_ok": False,
					"validation_ok": False,
					"schema_drift": True,
				},
			}
			return (self._format_mirror_markdown(fallback), False, fallback)

		expected = self._mirror_expected_keys()
		actual = set(obj.keys())
		extra_keys = sorted(list(actual - expected))
		missing_keys = sorted(list(expected - actual))
		meta: dict[str, Any] = {
			"parse_ok": True,
			"schema_drift": bool(extra_keys or missing_keys),
			"extra_keys": extra_keys,
			"missing_keys": missing_keys,
			"validation_ok": False,
		}

		normalized: dict[str, Any] = {
			"topic": str(obj.get("topic") or ontology.topic or "General").strip() or (ontology.topic or "General"),
			"message": str(obj.get("message") or "").strip(),
			"strengths": self._normalize_string_list(obj.get("strengths"), limit=3),
			"gaps": self._normalize_string_list(obj.get("gaps"), limit=5),
			"red_flags": self._normalize_string_list(obj.get("red_flags"), limit=3),
			"likely_followups": self._normalize_string_list(obj.get("likely_followups"), limit=3),
			"upgrade_lines": self._normalize_string_list(obj.get("upgrade_lines"), limit=2),
			"confidence": self._normalize_confidence(obj.get("confidence")),
		}

		# Strict schema validation (after our normalization). This prevents runtime crashes and enforces bounds.
		try:
			report_model = MirrorReport.model_validate(normalized)
			normalized = report_model.model_dump()
			meta["validation_ok"] = True
		except Exception as e:
			meta["validation_error"] = (str(e) or "validation failed")[:400]
			fallback = {
				"topic": ontology.topic or "General",
				"message": "Low confidence: the Mirror report was invalid. Ask a clarifying follow-up and retry.",
				"strengths": [],
				"gaps": ["Could not validate a structured Mirror report. Please retry."],
				"red_flags": [],
				"likely_followups": [],
				"upgrade_lines": [],
				"confidence": 0.2,
				"_meta": meta,
			}
			return (self._format_mirror_markdown(fallback), False, fallback)

		# Capture the model-reported confidence for gating rewrites (before calibration).
		try:
			meta["llm_confidence"] = float(normalized.get("confidence", 0.5))
		except Exception:
			meta["llm_confidence"] = float(self._normalize_confidence(normalized.get("confidence")))

		# If the model drifted from the schema, downgrade confidence and force safe behavior.
		if meta.get("schema_drift"):
			normalized["confidence"] = min(float(normalized.get("confidence", 0.5)), 0.35)
			if not (normalized.get("message") or "").strip():
				normalized["message"] = (
					"Low confidence: the Mirror output was incomplete. "
					"Please add 1–2 concrete details (constraints, scale, trade-offs) and retry."
				)

		normalized["_meta"] = meta

		# Quality rewrites: key off the model-reported confidence (pre-calibration),
		# but skip if the schema drifted (incomplete/unsafe to rewrite).
		llm_conf_for_rewrites = float(meta.get("llm_confidence") or 0.0)
		if (not meta.get("schema_drift")) and llm_conf_for_rewrites >= 0.4:
			# Run rewrite and soften in parallel (independent LLM calls)
			topic_str = str(normalized.get("topic") or "General")

			async def _do_rewrite() -> list[str]:
				return await self._rewrite_upgrade_lines_if_needed(
					question=q, user_answer=ua, topic=topic_str,
					upgrade_lines=list(normalized.get("upgrade_lines") or []),
					api_key=api_key,
				)

			async def _do_soften() -> list[str]:
				return await self._soften_red_flags_if_needed(
					question=q, user_answer=ua, topic=topic_str,
					red_flags=list(normalized.get("red_flags") or []),
					api_key=api_key,
				)

			meta["upgrade_rewrite_attempted"] = True
			meta["red_flags_soften_attempted"] = True
			try:
				rewritten, softened = await asyncio.gather(
					_do_rewrite(), _do_soften(), return_exceptions=True,
				)
				if isinstance(rewritten, list):
					normalized["upgrade_lines"] = rewritten
					meta["upgrade_rewrite_ok"] = True
				else:
					meta["upgrade_rewrite_ok"] = False
				if isinstance(softened, list):
					normalized["red_flags"] = softened
					meta["red_flags_soften_ok"] = True
				else:
					meta["red_flags_soften_ok"] = False
			except Exception:
				meta["upgrade_rewrite_ok"] = False
				meta["red_flags_soften_ok"] = False

		# Deterministic confidence calibration (do not rely solely on model self-report)
		try:
			cal_c, cal_meta = self._calibrate_mirror_confidence(
				llm_confidence=float(meta.get("llm_confidence") or normalized.get("confidence", 0.5)),
				ontology=ontology,
				question=q,
				user_answer=ua,
				strengths=list(normalized.get("strengths") or []),
				gaps=list(normalized.get("gaps") or []),
				schema_drift=bool(meta.get("schema_drift")),
			)
			normalized["confidence"] = cal_c
			meta.update({"confidence_calibrated": True, "confidence_details": cal_meta})
		except Exception:
			# Never fail Mirror due to calibration.
			meta["confidence_calibrated"] = False

		normalized["_meta"] = meta

		# Enforce gap/follow-up policies before applying low-confidence guard
		normalized = self._enforce_mirror_gap_and_followup_policy(normalized, ontology, ua)

		normalized = self._apply_low_confidence_guard(normalized)
		return (self._format_mirror_markdown(normalized), False, normalized)

	async def generate_mirror_report(
		self,
		*,
		question: str,
		user_answer: str,
		depth: Optional[str] = None,
		api_key: Optional[str] = None,
	) -> tuple[str, bool]:
		"""Backward-compatible wrapper for callers that only need markdown."""
		md, truncated, _ = await self.generate_mirror_report_structured(
			question=question,
			user_answer=user_answer,
			depth=depth,
			api_key=api_key,
		)
		return (md, truncated)

	def _style_overrides(self, style_mode: Optional[str], tone: Optional[str], layout: Optional[str], variability: Optional[float], seed: Optional[int]) -> str:
		"""Construct style and tone overrides for varied, professional outputs."""
		# Default behavior: do NOT inject extra style instructions.
		# Product wants predictable output; style customization is opt-in via parameters.
		if style_mode is None and tone is None and layout is None and variability is None and seed is None:
			return ""

		import random
		rng = random.Random(seed)
		v = 0.0 if variability is None else max(0.0, min(1.0, variability))

		# Presets
		presets: dict[str, str] = {
			"concise": "Keep it tight. 4–6 bullets max. Avoid subheadings unless necessary.",
			"deep-dive": "Provide rich sections with 'Why it matters', 'Trade-offs', and a short example.",
			"mentor": "Use a coaching voice. Add 'Pitfalls' and 'What to practice' sections when helpful.",
			"executive": "Lead with outcomes and business impact. Use short paragraphs and a 'Bottom line' section.",
			"faq": "Answer as an FAQ: 4–6 Q→A pairs covering the topic succinctly.",
			"qa": "Use a Q→A dialogue style for key points, then a brief summary.",
			"checklist": "Present an actionable checklist with clear steps and acceptance criteria.",
			"narrative": "Explain as a narrative walkthrough with sections 'Context → Decision → Result'.",
			"varied": "Choose any of: concise, deep-dive, mentor, executive, faq, qa, checklist, narrative based on question type.",
		}

		chosen_mode = (style_mode or "auto").lower()
		if chosen_mode in ("auto", "varied"):
			# Soft randomization by question type
			candidates = ["concise", "deep-dive", "mentor", "executive", "faq", "qa", "checklist", "narrative"]
			if v > 0:
				chosen_mode = rng.choice(candidates)
			else:
				chosen_mode = "executive"  # sensible default

		tone_map: dict[str, str] = {
			"neutral": "Neutral, precise, professional.",
			"friendly": "Warm, approachable, but still professional.",
			"mentor": "Supportive, coaching tone with practical tips.",
			"executive": "Crisp, outcome-focused, confident.",
			"academic": "Formal, rigorous definitions and citations where appropriate.",
			"coaching": "Encouraging, step-by-step guidance.",
		}
		tone_rule = tone_map.get((tone or "").lower(), "Neutral, precise, professional.")

		layout_map: dict[str, str] = {
			"bullets": "Prefer bullets with minimal headings.",
			"narrative": "Short paragraphs, minimal headings.",
			"qa": "Q→A pairs.",
			"faq": "FAQ format.",
			"checklist": "Checklist of steps.",
			"pros-cons": "Pros/Cons section included.",
			"steps": "Numbered steps first, details later.",
		}
		layout_rule = layout_map.get((layout or "").lower(), "Use judgement for best readability.")

		preset_rule = presets.get(chosen_mode, "")

		return (
			"\n\nStyle & Tone Overrides:"
			f"\n- Tone: {tone_rule}"
			f"\n- Layout preference: {layout_rule}"
			f"\n- Style preset: {chosen_mode} — {preset_rule}"
			"\n- Vary headings and bullet density to avoid repetitive structure; choose the lightest structure that conveys clarity."
			"\n- Do not force the earlier template sections if brevity or narrative works better for this question."
		)

	def _needs_conversation_context(self, question: str, previous_qna: Optional[List[Dict[str, str]]]) -> bool:
		"""
		Intelligently determine if this question actually needs conversation history.
		This dramatically reduces token usage by only including context when necessary.

		Uses word-boundary matching for pronouns (avoids 'that' matching inside
		'What is **that** design pattern') and multi-word phrases for follow-up
		indicators (avoids standalone questions like 'Compare TCP vs UDP' from
		being treated as follow-ups).
		"""
		if not previous_qna:
			return False
		
		question_lower = question.lower().strip()
		words = set(question_lower.split())
		
		# Multi-word phrases: strong follow-up signals (low false-positive)
		phrase_indicators = [
			'explain more', 'tell me more', 'elaborate on', 'clarify',
			'what about', 'how about', 'what if',
			'you said', 'you mentioned', 'earlier', 'before', 'previous',
			'the code', 'the solution', 'the example', 'the approach',
			'can you show', 'show me',
		]
		for phrase in phrase_indicators:
			if phrase in question_lower:
				return True
		
		# Check if question is very short (2 words or less — likely a follow-up like "and?" "yes" "no why?")
		if len(question.split()) <= 2:
			return True
		
		# Single-word pronoun check: only flag as follow-up if the question
		# is SHORT (≤8 words) AND contains a demonstrative pronoun at the start.
		# This avoids flagging "What is that design pattern called?" as a follow-up.
		if len(question.split()) <= 8:
			first_words = question_lower.split()[:3]
			start_pronouns = {'this', 'that', 'these', 'those'}
			if any(w in start_pronouns for w in first_words):
				return True
		
		# Explicit continuation words (must be a whole word, not substring)
		continuation_words = {'also', 'additionally', 'furthermore', 'moreover'}
		if words & continuation_words:
			return True
		
		# Otherwise, it's a standalone question - no context needed!
		return False

	def _build_conversation_context(self, previous_qna: List[Dict[str, str]], max_turns: int = 3) -> str:
		"""
		Build conversation context from previous Q&A pairs.
		REDUCED to last 3 turns (was 5) to save tokens.
		Only called when _needs_conversation_context returns True.
		"""
		if not previous_qna:
			return ""
		
		# Take only the last N turns (REDUCED from 5 to 3)
		recent_qna = previous_qna[-max_turns:] if len(previous_qna) > max_turns else previous_qna
		
		context_parts = [
			"## CONVERSATION HISTORY",
			"",
			"Previous questions and answers in this session:",
			""
		]
		
		for i, item in enumerate(recent_qna, 1):
			q = item.get('question', '').strip()
			a = item.get('answer', '').strip()
			
			# Truncate answer more aggressively (300 chars instead of 500)
			if len(a) > 300:
				a = a[:300] + "..."
			
			context_parts.append(f"**Q{i}:** {q}")
			context_parts.append(f"**A{i}:** {a}")
			context_parts.append("")
		
		context_parts.append("---")
		context_parts.append("")
		
		return "\n".join(context_parts)

	def _build_prompt_with_profile(
		self,
		question: str,
		system_prompt: Optional[str],
		profile_text: Optional[str],
		*,
		depth: Optional[str] = None,
	) -> str:
		"""
		Shared helper to build prompt with profile context using world-class analyzer
		
		This eliminates code duplication between generate_answer() and stream_answer()
		"""
		if system_prompt is not None:
			# If an external system_prompt is provided, append the product-critical
			# response contract + hygiene so output structure stays consistent.
			prompt = (
				system_prompt
				+ "\n\n"
				+ "\n\n".join([
					PROMPT_INJECTION_RESISTANCE.text,
					RESPONSE_TEMPLATE.text,
					OUTPUT_GUARDS.text,
					RESPONSE_CONTRACT.text,
					DEPTH_BUDGET.text,
					OUTPUT_HYGIENE.text,
				])
			)
		else:
			app_name, developer, attribution = self._get_app_identity()
			plan = self._build_response_plan(question, depth=depth)
			flags = self._flags_from_plan(plan, question)
			prompt = build_default_system_prompt(
				app_name=app_name,
				developer_name=developer,
				attribution=attribution,
				flags=flags,
			)
		
		if not profile_text:
			return prompt

		# Guard: truncate excessively large profile text to prevent context-window blowout.
		MAX_PROFILE_CHARS = 4000
		safe_text = profile_text.strip()
		if len(safe_text) > MAX_PROFILE_CHARS:
			safe_text = safe_text[:MAX_PROFILE_CHARS] + "\n…[truncated – profile too long]"
			logger.warning(
				f"Profile text truncated from {len(profile_text)} to {MAX_PROFILE_CHARS} chars"
			)

		# Simple profile context injection (no special document analysis behavior).
		prompt = (
			prompt
			+ "\n\n"
			+ "=== Candidate Profile Context (Use for resume/personal questions) ===\n"
			+ safe_text
			+ "\n=== End of Profile ===\n"
		)
		if self._needs_first_person(question):
			prompt = prompt + self._persona_overrides()
		
		return prompt

	def _build_full_prompt(
		self,
		question: str,
		system_prompt: Optional[str],
		profile_text: Optional[str],
		previous_qna: Optional[List[Dict[str, str]]],
		apply_auto_overrides: bool,
		*,
		depth: Optional[str] = None,
		style_mode: Optional[str] = None,
		tone: Optional[str] = None,
		layout: Optional[str] = None,
		variability: Optional[float] = None,
		seed: Optional[int] = None,
	) -> str:
		"""Build complete prompt with base + history + overrides + style.
		
		Consolidates duplicate logic from generate_answer() and stream_answer().
		"""
		# 1. Build base prompt with profile context
		base_prompt = self._build_prompt_with_profile(question, system_prompt, profile_text, depth=depth)
		
		# 2. Conversation history is handled via message roles (user/assistant) in _build_messages.
		# Avoid duplicating it inside the system prompt to reduce token bloat and role confusion.
		if previous_qna and self._needs_conversation_context(question, previous_qna):
			logger.info(f"💬 Including conversation context ({len(previous_qna)} turns available, using last 3)")
		else:
			logger.info(f"⚡ Standalone question - skipping conversation history (saving tokens!)")
		
		prompt = base_prompt
		
		# 3. Apply auto-overrides if enabled
		if apply_auto_overrides:
			if self._is_identity_question(question):
				prompt = prompt + self._identity_overrides()
			else:
				if self._needs_comparison(question):
					prompt = prompt + self._comparison_overrides(question)
				if self._is_greeting(question):
					prompt = prompt + self._greeting_overrides()
				if self._is_off_topic(question):
					prompt = prompt + self._off_topic_overrides()
				if self._is_ambiguous(question):
					prompt = prompt + self._ambiguous_query_overrides()
				if not self._has_sufficient_context(question, previous_qna):
					prompt = prompt + self._context_fallback_overrides()
		else:
			logger.debug("🧪 Auto overrides disabled for internal call")
		
		# 4. Apply style & tone overrides
		prompt = prompt + self._style_overrides(style_mode, tone, layout, variability, seed)
		
		return prompt

	def _build_messages(
		self,
		question: str,
		prompt: str,
		previous_qna: Optional[List[Dict[str, str]]] = None,
	) -> List[Dict[str, str]]:
		"""Build message list for LLM with optional conversation history.
		
		Consolidates duplicate logic from generate_answer() and stream_answer().
		"""
		messages: List[Dict[str, str]] = [
			{"role": "system", "content": prompt}
		]
		
		# Add conversation history if relevant
		if previous_qna and self._needs_conversation_context(question, previous_qna):
			for item in previous_qna[-3:]:
				q = (item.get("question") or "").strip()
				a = (item.get("answer") or "").strip()
				if q:
					messages.append({"role": "user", "content": q})
				if a:
					messages.append({"role": "assistant", "content": a})
		
		# Current question last
		messages.append({"role": "user", "content": question})
		return messages

	def _render_messages_as_text_transcript(self, messages: List[Dict[str, str]]) -> str:
		"""Render role-based messages into a single text prompt.

		Used for providers that don't accept role-separated chat messages.
		"""
		if not messages:
			return ""
		parts: list[str] = []
		for msg in messages:
			role = (msg.get("role") or "").strip().lower()
			content = (msg.get("content") or "").strip()
			if not content:
				continue
			if role == "system":
				parts.append(content)
			elif role == "user":
				parts.append(f"User: {content}")
			elif role == "assistant":
				parts.append(f"Assistant: {content}")
			else:
				parts.append(f"{role.title()}: {content}")
		return "\n\n".join(parts).strip()

	async def generate_answer(
		self,
		question: str,
		system_prompt: Optional[str] = None,
		profile_text: Optional[str] = None,
		previous_qna: Optional[List[Dict[str, str]]] = None,
		*,
		depth: Optional[str] = None,
		style_mode: Optional[str] = None,
		tone: Optional[str] = None,
		layout: Optional[str] = None,
		variability: Optional[float] = None,
		seed: Optional[int] = None,
		api_key: Optional[str] = None,
		apply_auto_overrides: bool = True,
		allow_provider_fallback: bool = True,
		groq_model_override: Optional[str] = None,
		restrict_groq_to_override: bool = False,
		user_tier: str = "standard",
	) -> tuple[str, bool]:
		# Deterministic identity answers (avoid LLM hallucinated attribution)
		if self._is_identity_question(question):
			logger.info("🪪 [IDENTITY] generate_answer short-circuit: %s", (question or "")[:200])
			return (self._identity_response_text(question), False)

		# Only consider demo key pool when it's explicitly enabled in settings.
		if settings.is_demo_key_pool_enabled():
			pool_keys = set(demo_key_pool.keys())
		else:
			pool_keys = set()
		attempt_key = api_key
		last_err: Exception | None = None

		# Build complete prompt using unified helper
		prompt = self._build_full_prompt(
			question,
			system_prompt,
			profile_text,
			previous_qna,
			apply_auto_overrides,
			depth=depth,
			style_mode=style_mode,
			tone=tone,
			layout=layout,
			variability=variability,
			seed=seed,
		)

		temperature = self._settings.answer_temperature
		top_p = self._settings.groq_top_p
		max_tokens = self._get_optimal_token_limit(question, self._settings.groq_max_tokens, depth=depth, user_tier=user_tier)
		stream = self._settings.groq_stream
		# When we detect provider truncation, we may need a higher output cap for continuation.
		max_tokens_ceiling = (
			self._settings.groq_max_tokens
			or getattr(self._settings, "groq_max_tokens_complex", None)
			or max_tokens
		)
		max_tokens_ceiling = int(max(max_tokens_ceiling, max_tokens))

		def build_kwargs(stream_flag: bool, model_name: str, *, messages_override: Optional[list[dict[str, str]]] = None, max_tokens_override: Optional[int] = None):
			# Build messages using unified helper
			messages = messages_override if messages_override is not None else self._build_messages(question, prompt, previous_qna)
			kwargs = {
				"model": model_name,
				"messages": messages,
				"temperature": temperature,
				"max_tokens": int(max_tokens_override if max_tokens_override is not None else max_tokens),
			}
			if top_p is not None:
				kwargs["top_p"] = top_p
			if stream_flag:
				kwargs["stream"] = True
			return kwargs

		def _build_continuation_messages(
			*,
			assistant_tail: str,
		) -> list[dict[str, str]]:
			base_messages = self._build_messages(question, prompt, previous_qna)
			# Include only the tail to keep context bounded.
			tail = (assistant_tail or "").strip()
			if tail:
				base_messages.append({"role": "assistant", "content": tail})
			base_messages.append(
				{
					"role": "user",
					"content": (
						"Continue from EXACTLY where you stopped. "
						"Do NOT repeat earlier content. "
						"If you were mid-bullet or inside a code block, resume correctly."
					),
				}
			)
			return base_messages

		async def _call(client_local, provider_local) -> tuple[str, bool]:
			"""Returns (answer, truncated) with automated fallback for rate limits"""
			client = client_local
			provider = provider_local
			# Provider-specific logic:
			# - Groq: try primary + fallback models
			# - Gemini: single call (no Groq model list)

			async def _call_groq(groq_client) -> tuple[str, bool]:
				for current_model in self._groq_models_to_try(
					groq_model_override=groq_model_override,
					restrict_to_override=restrict_groq_to_override,
				):
					try:
						# First pass (may stream based on settings)
						if stream:
							stream_resp = groq_client.chat.completions.create(**build_kwargs(True, current_model))
							parts: list[str] = []
							finish_reason = None
							for chunk in stream_resp:
								parts.append(getattr(chunk.choices[0].delta, "content", None) or "")
								if hasattr(chunk.choices[0], "finish_reason") and chunk.choices[0].finish_reason:
									finish_reason = chunk.choices[0].finish_reason
							raw_chunks: list[str] = [("".join(parts) or "").strip()]
						else:
							resp = groq_client.chat.completions.create(**build_kwargs(False, current_model))
							raw_chunks = [((resp.choices[0].message.content or "").strip())]
							finish_reason = resp.choices[0].finish_reason

						truncated = (finish_reason == "length")
						# Auto-continue on truncation (bounded to avoid runaway costs)
						continuations = 0
						while truncated and continuations < 2:
							assistant_tail = ("\n".join(raw_chunks))[-4000:]
							cont_messages = _build_continuation_messages(assistant_tail=assistant_tail)
							cont_resp = groq_client.chat.completions.create(
								**build_kwargs(
									False,
									current_model,
									messages_override=cont_messages,
									max_tokens_override=max_tokens_ceiling,
								)
							)
							cont_text = ((cont_resp.choices[0].message.content or "").strip())
							if cont_text:
								raw_chunks.append(cont_text)
							finish_reason = cont_resp.choices[0].finish_reason
							truncated = (finish_reason == "length")
							continuations += 1

						raw_text = "\n".join([c for c in raw_chunks if c]).strip()
						formatted = self._format_response(raw_text)
						return (formatted, truncated)
					except Exception as e:
						error_msg = str(e).lower()
						should_retry = any(
							x in error_msg
							for x in ["429", "rate_limit", "400", "decommissioned", "not found", "invalid_request_error"]
						)
						if should_retry:
							logger.warning(f"⚠️ Model {current_model} failed ({e}). Trying next fallback...")
							continue
						raise
				raise Exception("All Groq models reached their limits or failed.")

			async def _call_gemini(gemini_client) -> tuple[str, bool]:
				import anyio
				gmodel = gemini_client.GenerativeModel(self._settings.gemini_model)
				messages = self._build_messages(question, prompt, previous_qna)
				full_prompt = self._render_messages_as_text_transcript(messages)
				resp = await anyio.to_thread.run_sync(gmodel.generate_content, full_prompt)
				raw_text = getattr(resp, "text", None) or (
					resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else ""
				)
				formatted = self._format_response((raw_text or "").strip())
				return (formatted, False)

			# Primary provider call
			try:
				if provider == "groq":
					return await _call_groq(client)
				if provider == "gemini":
					return await _call_gemini(client)
			except Exception as primary_err:
				# Some flows (e.g. demo) must not fall back across providers.
				if not allow_provider_fallback:
					raise
				# Symmetric fallback across providers
				if provider == "groq":
					try:
						gemini_client, _ = self._ensure_client_by_provider("gemini")
						if gemini_client:
							logger.info("🔄 Groq failed. Falling back to Gemini...")
							return await _call_gemini(gemini_client)
					except Exception as gemini_err:
						logger.error(f"❌ Gemini fallback also failed: {gemini_err}")
						raise Exception("All LLM providers and models reached their limits or failed.") from primary_err
				elif provider == "gemini":
					try:
						groq_client, _ = self._ensure_client_by_provider("groq")
						if groq_client:
							logger.info("🔄 Gemini failed. Falling back to Groq...")
							return await _call_groq(groq_client)
					except Exception as groq_err:
						logger.error(f"❌ Groq fallback also failed: {groq_err}")
						raise Exception("All LLM providers and models reached their limits or failed.") from primary_err
				raise

			raise Exception("All LLM providers and models reached their limits or failed.")

		for attempt in range(2):
			client, provider = self._ensure_client(attempt_key)
			if client is None:
				return (question, False)  # mock: echo when no key, not truncated
			try:
				formatted, truncated = await _call(client, provider)
				# Post-process formatting: ensure definition questions get a 3-bullet
				# summary if model failed to emit bullets, and strip side headings
				# for very short answers.
				# Determine if the question appears to be a definition request.
				if self._is_definition_question(question):
					formatted = self._ensure_three_bullet_summary(formatted, question)
				# For short answers, remove sub-headings introduced by the model.
				formatted = self._strip_side_headings_for_short(formatted)

				# Auto-append short recommendations for concept/definition answers if missing
				# (guarded: helper methods may be absent in this branch)
				if (
					self._should_auto_append_recommendations(question, formatted)
				):
					recs = self._auto_recommendations_for_question(question, formatted)
					if recs:
						formatted = self._append_recommendations_block(formatted, recs)
				return (formatted, truncated)
			except Exception as e:
				last_err = e
				if (
					attempt == 0
					and attempt_key
					and attempt_key in pool_keys
					and _is_quota_or_rate_limit_error(e)
				):
					demo_key_pool.mark_exhausted(attempt_key, reason=str(e)[:200])
					attempt_key = demo_key_pool.get_key()  # rotate
					if attempt_key:
						logger.warning("[DEMO_KEY_POOL] Retrying generate_answer with a different demo key")
						continue
				raise

		# Should never reach here, but keeps mypy/linters happy.
		raise last_err if last_err else Exception("LLM call failed")

	def _is_definition_question(self, q: str) -> bool:
		# Stricter detection: only treat as definition when the user explicitly
		# asks for a definition or the question is short and looks definitional.
		import re
		q_raw = (q or "").strip()
		q = q_raw.lower()
		# Explicit prefixes like 'What is X' or 'Define X' or 'Definition of X'
		if re.match(r"^(what\s+is|define|definition\s+of|meaning\s+of)\b", q):
			return True
		# Short form: "X: definition" or "X meaning" (very short queries)
		if len(q_raw.split()) <= 5 and any(k in q for k in [" meaning", " definition", "defined as"]):
			return True
		return False

	def _ensure_three_bullet_summary(self, text: str, question: str) -> str:
		"""If the question is a definition-type and the model didn't emit bullets,
		build a concise 3-item bulleted summary.
		Try to reuse sentences from the model output when possible.
		"""
		if not text or not self._is_definition_question(question):
			return text
		# If already contains bullets, assume model complied
		if "\n- " in text or "\n* " in text:
			return text
		# Extract up to three short sentences from the model output to populate bullets
		import re
		sents = [s.strip() for s in re.split(r'[\n\.\?!]+', text) if s.strip()]
		first = sents[0] if len(sents) > 0 else ""
		second = sents[1] if len(sents) > 1 else ""
		third = sents[2] if len(sents) > 2 else ""
		# Build bullets with fallbacks (no labeled headings; keep it natural)
		definition = first or "A concise definition of the concept."
		why = second or "Why it matters in practice."
		example = third or "A short concrete example or use case."
		return f"- {definition}\n- {why}\n- {example}"

	def _strip_side_headings_for_short(self, text: str) -> str:
		"""For short answers, remove sub/headings to keep output compact.
		If the whole text is under ~40 words and contains bolded or 'Details:' style
		headings, strip those heading markers and join the content.
		"""
		if not text:
			return text
		word_count = len(text.split())
		# Only transform very short answers (keep conservatively small)
		if word_count > 25:
			return text
		# Don't strip if bullets or lists are present (model intended structure)
		if "\n- " in text or "\n* " in text or "\n1." in text:
			return text
		import re
		# Remove 'Details:', 'Concrete example:' style labels at line starts
		text = re.sub(r'(?m)^(?:\*\*?\s*[^\n:\*]{1,60}\*\*?\s*:\s*|Details:\s*|Concrete example:\s*)', '', text)
		# Remove markdown headings like '## Heading' or '**Heading**' when short
		text = re.sub(r'(?m)^#{1,3}\s*', '', text)
		text = re.sub(r'\*\*(.*?)\*\*', r"\1", text)
		# Collapse multiple blank lines
		text = re.sub(r"\n{2,}", "\n\n", text).strip()
		return text

	def _has_recommendations_already(self, text: str) -> bool:
		"""Detect whether the answer already includes a recommendations/next-steps style block.

		We keep this lightweight and deterministic because it's used in hot paths
		(including streaming).
		"""
		t = (text or "").lower()
		if not t:
			return False
		markers = (
			"recommendations",
			"recommended",
			"next steps",
			"what to do next",
			"resources",
			"further reading",
			"practice",
			"try this",
		)
		return any(m in t for m in markers)

	def _should_auto_append_recommendations(self, question: str, answer_text: str) -> bool:
		"""Decide when to add a tiny 'next steps' block.

		Root cause of inconsistency: we previously relied on optional methods that
		didn't always exist and also on the model sometimes adding a section by itself.
		"""
		q = (question or "").strip().lower()
		a = (answer_text or "").strip()
		if not q or not a:
			return False
		# Only for definition/concept questions (to avoid bloating all answers).
		if not self._is_definition_question(question):
			return False
		# Don't add if the model already included something similar.
		if self._has_recommendations_already(a):
			return False
		# Avoid spamming very short answers or extremely long outputs.
		wc = len(a.split())
		if wc < 20:
			return False
		if wc > 900:
			return False
		# If user asked for "only" a definition, respect that.
		if any(k in q for k in ["only definition", "just definition", "only the definition", "one line definition", "one-line definition"]):
			return False
		return True

	def _auto_recommendations_for_question(self, question: str, answer_text: str) -> list[str]:
		"""Generate topic-aware next steps based on the question content.

		No LLM calls here on purpose (cost + consistency).
		"""
		q = (question or "").strip()
		a = (answer_text or "").strip()
		if not q or not a:
			return []

		q_lower = q.lower()

		# System design questions
		if any(t in q_lower for t in ("system design", "design a", "architect", "scalab", "distributed")):
			return [
				"Sketch the data model and identify the hottest read/write paths.",
				"Prepare to discuss a concrete scaling bottleneck and how you'd resolve it.",
				"Practice explaining trade-offs (consistency vs availability, cost vs latency).",
			]

		# Coding / algorithm questions
		if any(t in q_lower for t in ("algorithm", "implement", "code", "function", "leetcode", "time complexity", "big o")):
			return [
				"Run through at least 2 edge cases (empty input, single element, duplicates).",
				"Be ready to explain your time and space complexity clearly.",
				"Think about an alternative approach and why you chose this one.",
			]

		# Database / SQL questions
		if any(t in q_lower for t in ("database", "sql", "schema", "index", "query", "normalization", "nosql")):
			return [
				"Know when to normalize vs denormalize and the trade-offs of each.",
				"Prepare to discuss indexing strategy for the most common query pattern.",
				"Practice explaining your schema decisions in under 60 seconds.",
			]

		# Behavioral questions
		if any(t in q_lower for t in ("tell me about a time", "behavioral", "leadership", "conflict", "teamwork", "challenge")):
			return [
				"Structure your answer using STAR (Situation, Task, Action, Result).",
				"Quantify the result if possible (saved X hours, improved Y%).",
				"Prepare a 30-second version and a 2-minute version of this story.",
			]

		# Default: general interview prep
		return [
			"Know 1 real-world use case and 1 common pitfall.",
			"Be ready to compare it with the closest alternative (trade-offs).",
			"Practice a 30-second explanation, then a 2-minute deep version.",
		]

	def _append_recommendations_block(self, text: str, recommendations: list[str]) -> str:
		if not text:
			return text
		recs = [r.strip() for r in (recommendations or []) if str(r).strip()]
		if not recs:
			return text
		# Keep formatting consistent with the rest of the app.
		block = "\n\n**Next steps**\n" + "\n".join([f"- {r}" for r in recs[:3]])
		return (text.rstrip() + block + "\n")

	async def evaluate_code_with_critique(self, problem: str, code: str, language: str, conversation_context: str = "", api_key: Optional[str] = None) -> str:
		"""Ask the model to produce a structured evaluation and approach explanation."""
		client, provider = self._ensure_client(api_key)
		if client is None:
			raise Exception("LLM client not available.")

		system = (
			"You are an expert technical interviewer. Evaluate the following code solution.\n\n"
			"Structure your response EXACTLY as follows:\n"
			"Summary: [Brief 2-3 sentence overview]\n\n"
			"Approach: [Detailed explanation of the algorithm and data structures used]\n\n"
			"Strengths:\n- [Strength 1]\n- [Strength 2]\n\n"
			"Weaknesses:\n- [Weakness 1]\n- [Weakness 2]\n\n"
			"Recommendations:\n- [Actionable improvement 1]\n- [Actionable improvement 2]\n\n"
			"Scores: {\"correctness\": score, \"optimization\": score, \"approach_explanation\": score, \"complexity_discussion\": score, \"edge_cases_testing\": score, \"total\": score}\n\n"
			"Use a professional, encouraging but critical tone. Provide deep technical insights."
		)

		user_prompt = f"Problem Context: {problem}\n\nCode to Evaluate ({language}):\n```\n{code}\n```\n\nPrevious Conversation Context:\n{conversation_context}"

		async def _call():
			# Non-Groq providers don't need model fallback lists.
			if provider != "groq":
				gmodel = client.GenerativeModel(self._settings.gemini_model.replace("models/", ""))
				resp = gmodel.generate_content(f"{system}\n\n{user_prompt}")
				return getattr(resp, "text", "")

			for current_model in self._groq_models_to_try():
				try:
					resp = client.chat.completions.create(
						model=current_model,
						messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
						temperature=0.2,
						max_tokens=self._settings.groq_max_tokens_complex,
					)
					return resp.choices[0].message.content
				except Exception as e:
					error_msg = str(e).lower()
					should_retry = any(x in error_msg for x in ["429", "rate_limit", "400", "decommissioned", "not found", "invalid_request_error"])
					
					if should_retry:
						logger.warning(f"⚠️ [EVAL] Model {current_model} failed ({e}). Trying next fallback...")
						continue
					raise e
			
			# Last resort fallback to Gemini if Groq exhausted
			if provider == "groq":
				try:
					gemini_client, _ = self._ensure_client_by_provider("gemini")
					if gemini_client:
						gmodel = gemini_client.GenerativeModel(self._settings.gemini_model.replace("models/", ""))
						resp = gmodel.generate_content(f"{system}\n\n{user_prompt}")
						return getattr(resp, "text", "")
				except Exception:
					pass
			
			raise Exception("Evaluation failed: Rate limit reached for all models.")

		return await _call()

	async def classify_is_technical(self, question: str, answer: str, api_key: Optional[str] = None) -> tuple[bool, float, str]:
		"""
		Classify if content is technical/evaluatable using fast keyword heuristics.
		No LLM call needed — code blocks, technical keywords, and question patterns
		are sufficient for routing decisions.
		"""
		q_lower = (question or "").lower()
		a_lower = (answer or "").lower()
		combined = q_lower + " " + a_lower

		# 1. Structural: code blocks in question or answer → definitely technical
		if "```" in answer or "```" in question:
			return True, 1.0, "code block detected"

		# 2. Keyword sets for technical categories
		_CODING_TERMS = {
			"algorithm", "function", "class", "method", "array", "hashmap", "hash map",
			"linked list", "binary tree", "stack", "queue", "recursion", "loop",
			"pointer", "variable", "compile", "runtime", "exception", "debug",
			"api", "endpoint", "rest", "graphql", "http", "tcp", "udp",
			"sorting", "searching", "dynamic programming", "dfs", "bfs",
			"big o", "time complexity", "space complexity", "o(n)", "o(1)",
			"python", "javascript", "java", "typescript", "golang", "rust", "c++",
			"react", "node", "django", "flask", "fastapi", "spring",
		}
		_SYSTEM_DESIGN_TERMS = {
			"system design", "scalability", "load balancer", "microservice",
			"distributed", "cache", "caching", "redis", "kafka", "message queue",
			"sharding", "partition", "replication", "cdn", "latency",
			"throughput", "availability", "consistency", "cap theorem",
			"database design", "schema", "sql", "nosql", "mongodb", "postgresql",
			"indexing", "normalization", "denormalization", "acid",
		}
		_ARCHITECTURE_TERMS = {
			"architecture", "design pattern", "singleton", "factory", "observer",
			"mvc", "mvvm", "monolith", "serverless", "containerization",
			"docker", "kubernetes", "ci/cd", "deployment", "infrastructure",
		}

		all_terms = _CODING_TERMS | _SYSTEM_DESIGN_TERMS | _ARCHITECTURE_TERMS

		matched = [t for t in all_terms if t in combined]
		if len(matched) >= 2:
			category = "system_design" if any(t in _SYSTEM_DESIGN_TERMS for t in matched) else "coding"
			return True, 0.95, category
		if len(matched) == 1:
			return True, 0.8, "technical_keyword"

		# 3. Fallback: not technical
		return False, 0.1, "no technical signals"

	def _stream_sanitize_chunk(self, chunk: str) -> str:
		"""Lightweight per-chunk sanitizer for streaming responses.

		Applies the most visible fixes without needing full-document context:
		- Unicode bullet → hyphen bullet
		- Smart quotes → ASCII quotes
		- Prompt-leak lines removed
		"""
		if not chunk:
			return chunk
		# Unicode bullet markers → '- '
		for marker in ("•", "·", "‣", "◦"):
			chunk = chunk.replace(f"{marker} ", "- ")
			# Lone marker at start of chunk
			if chunk.startswith(marker):
				chunk = "- " + chunk[len(marker):].lstrip()
		# Smart quotes → ASCII
		chunk = chunk.replace("\u2019", "'").replace("\u2018", "'")
		chunk = chunk.replace("\u201c", '"').replace("\u201d", '"')
		chunk = chunk.replace("\u2013", "-").replace("\u2014", "-")
		chunk = chunk.replace("\u2026", "...")
		return chunk


	async def stream_answer(
		self,
		question: str,
		system_prompt: Optional[str] = None,
		profile_text: Optional[str] = None,
		previous_qna: Optional[List[Dict[str, str]]] = None,
		*,
		depth: Optional[str] = None,
		style_mode: Optional[str] = None,
		tone: Optional[str] = None,
		layout: Optional[str] = None,
		variability: Optional[float] = None,
		seed: Optional[int] = None,
		api_key: Optional[str] = None,
		apply_auto_overrides: bool = True,
		allow_provider_fallback: bool = True,
		groq_model_override: Optional[str] = None,
		restrict_groq_to_override: bool = False,
		user_tier: str = "standard",
	) -> AsyncIterator[str]:
		# Deterministic identity answers (avoid LLM hallucinated attribution)
		if self._is_identity_question(question):
			logger.info("🪪 [IDENTITY] stream_answer short-circuit: %s", (question or "")[:200])
			yield self._identity_response_text(question)
			return

		client, provider = self._ensure_client(api_key)

		if client is None:
			yield ""
			return

		# Build complete prompt using unified helper
		prompt = self._build_full_prompt(
			question,
			system_prompt,
			profile_text,
			previous_qna,
			apply_auto_overrides,
			depth=depth,
			style_mode=style_mode,
			tone=tone,
			layout=layout,
			variability=variability,
			seed=seed,
		)

		# Use dynamic token limit for streaming
		max_tokens = self._get_optimal_token_limit(question, self._settings.groq_max_tokens, depth=depth, user_tier=user_tier)
		max_tokens_ceiling = (
			self._settings.groq_max_tokens
			or getattr(self._settings, "groq_max_tokens_complex", None)
			or max_tokens
		)
		max_tokens_ceiling = int(max(max_tokens_ceiling, max_tokens))

		def _call_stream(current_model: str, *, messages_override: Optional[list[dict[str, str]]] = None, max_tokens_override: Optional[int] = None):
			# Build messages using unified helper
			messages = messages_override if messages_override is not None else self._build_messages(question, prompt, previous_qna)
			if provider == "groq":
				return client.chat.completions.create(
					model=current_model,
					messages=messages,
					temperature=self._settings.answer_temperature,
					max_tokens=int(max_tokens_override if max_tokens_override is not None else max_tokens),
					stream=True,
				)
			elif provider == "gemini":
				return None
			else:
				return None

		import anyio
		models_to_try = self._groq_models_to_try(
			groq_model_override=groq_model_override,
			restrict_to_override=restrict_groq_to_override,
		)
		stream = None
		active_provider = provider
		
		if provider == "groq":
			for current_model in models_to_try:
				try:
					stream = await anyio.to_thread.run_sync(_call_stream, current_model)
					break # Success
				except Exception as e:
					if "429" in str(e) or "rate_limit" in str(e).lower():
						logger.warning(f"⚠️ [STREAM] Model {current_model} rate limited. Trying fallback...")
						continue
					raise e
			
			# If everything failed, try Gemini last resort (unless disabled, e.g. Demo Mode)
			if stream is None and allow_provider_fallback:
				try:
					gemini_client, _ = self._ensure_client_by_provider("gemini")
					if gemini_client:
						active_provider = "gemini"
						client = gemini_client
						logger.info("🔄 [STREAM] Groq failed. Falling back to Gemini...")
				except Exception:
					pass
		else:
			stream = await anyio.to_thread.run_sync(_call_stream, self._settings.groq_model) # Should not be called if not groq

		if active_provider == "groq" and stream is not None:
			# Stream (with bounded auto-continue) to avoid mid-answer cutoffs.
			continuations = 0
			assistant_tail = ""
			saw_recommendations = False
			while True:
				finish_reason = None
				inside_think = False
				carry = ""
				for chunk in stream:
					piece = getattr(chunk.choices[0].delta, "content", None) or ""
					# Handle thinking tags in stream
					if "<think>" in piece:
						piece = piece.replace("<think>", "<details class='thinking-process'><summary>Thinking Process</summary>")
						inside_think = True
					if "</think>" in piece:
						piece = piece.replace("</think>", "</details>")
						inside_think = False
					if piece:
						combined = carry + piece
						parts = combined.splitlines(True)
						carry = ""
						if parts and not parts[-1].endswith(("\n", "\r")):
							carry = parts.pop()
						out_parts: list[str] = []
						for seg in parts:
							line = seg.rstrip("\r\n")
							if self._is_internal_prompt_leak_line(line):
								continue
							out_parts.append(seg)
						emit = "".join(out_parts)
						if emit:
							emit = self._stream_sanitize_chunk(emit)
							if (not saw_recommendations) and self._has_recommendations_already(emit):
								saw_recommendations = True
							assistant_tail = (assistant_tail + emit)[-4000:]
							yield emit
					# Track finish reason to detect truncation
					if hasattr(chunk.choices[0], "finish_reason") and chunk.choices[0].finish_reason:
						finish_reason = chunk.choices[0].finish_reason

				# Flush any remaining partial line (only if it isn't leakage)
				if carry and not self._is_internal_prompt_leak_line(carry):
					if (not saw_recommendations) and self._has_recommendations_already(carry):
						saw_recommendations = True
					assistant_tail = (assistant_tail + carry)[-4000:]
					yield carry

				if finish_reason != "length":
					break

				# Auto-continue once or twice when the provider hits output limit.
				if continuations >= 2:
					break
				continuations += 1
				cont_messages = self._build_messages(question, prompt, previous_qna)
				if assistant_tail:
					cont_messages.append({"role": "assistant", "content": assistant_tail.strip()})
				cont_messages.append(
					{
						"role": "user",
						"content": (
							"Continue from EXACTLY where you stopped. "
							"Do NOT repeat earlier content. "
							"If you were mid-bullet or inside a code block, resume correctly."
						),
					}
				)
				# Restart stream with higher cap to reduce repeated truncations.
				stream = await anyio.to_thread.run_sync(_call_stream, self._settings.groq_model, messages_override=cont_messages, max_tokens_override=max_tokens_ceiling)
				continue

			# If still truncated after continuations, append a small notice.
			if finish_reason == "length":
				yield "\n\n---\n\n**⚠️ Response Truncated:** The answer exceeded the output limit and was cut short. Ask to continue or narrow to a specific section."
			else:
				# Append deterministic next-steps for definition questions if missing.
				if (not saw_recommendations) and self._should_auto_append_recommendations(question, assistant_tail):
					recs = self._auto_recommendations_for_question(question, assistant_tail)
					if recs:
						block = "\n\n**Next steps**\n" + "\n".join([f"- {r}" for r in recs[:3]])
						yield block + "\n"
		elif active_provider == "gemini":
			# Non-streaming fallback: yield once
			async def _one_shot():
				try:
					gmodel = client.GenerativeModel(self._settings.gemini_model.replace("models/", ""))
					messages = self._build_messages(question, prompt, previous_qna)
					full_prompt = self._render_messages_as_text_transcript(messages)
					resp = await anyio.to_thread.run_sync(gmodel.generate_content, full_prompt)
					raw_text = getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "")
					formatted = self._format_response((raw_text or "").strip())
					# Mirror generate_answer post-processing for parity.
					if self._is_definition_question(question):
						formatted = self._ensure_three_bullet_summary(formatted, question)
					formatted = self._strip_side_headings_for_short(formatted)
					if self._should_auto_append_recommendations(question, formatted):
						recs = self._auto_recommendations_for_question(question, formatted)
						if recs:
							formatted = self._append_recommendations_block(formatted, recs)
					return formatted
				except Exception as e:
					logger.error(f"Gemini fallback error: {e}")
					return ""
			
			text_once = await _one_shot()
			if text_once:
				yield text_once



# Attach all text post-processing functions from response_postprocess as methods
# on LLMService so callers can use self._format_response(...) etc.
response_postprocess.attach_text_postprocess_methods(LLMService)

# Factory to get an LLMService for a specific provider (gemini/groq)
_llm_service_instances = {}
def get_llm_service(provider: str = None, feature: str = "default") -> LLMService:
	"""
	Returns an LLMService instance for the given provider ('gemini' or 'groq').
	If provider is None, intelligently selects provider based on available API keys:
	  - Only Groq key: use Groq for all features
	  - Only Gemini key: use Gemini for all features
	  - Both keys: use Gemini for Copilot Chat (feature="copilot"), Groq for others
	
	Args:
		provider: Explicit provider name ('groq' or 'gemini'), or None for auto-detection
		feature: Feature context ("copilot" for AI Copilot Chat, "default" for others)
		
	Caches instances per provider.
	"""
	from app.config import settings as _settings
	
	# If provider explicitly specified, use it
	if provider:
		key = provider.lower()
	else:
		# Intelligently determine provider based on available keys and feature
		key = _settings.get_effective_provider(feature)
	
	if key not in _llm_service_instances:
		# Patch settings for this instance
		import copy
		inst_settings = copy.deepcopy(_settings)
		inst_settings.llm_provider = key
		svc = LLMService()
		# Patch the settings used by this instance
		svc._settings = inst_settings
		_llm_service_instances[key] = svc
	return _llm_service_instances[key]

# Default global instance (uses intelligent provider selection)
llm_service = get_llm_service()
