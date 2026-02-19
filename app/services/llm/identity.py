from __future__ import annotations

import re


def get_app_identity(settings) -> tuple[str, str, str]:
	"""Returns (app_name, developer_name, attribution)."""
	return (
		settings.app_name,
		settings.app_developer_name,
		settings.app_developer_attribution,
	)


def app_name_targets(settings) -> list[str]:
	"""Lowercase identity targets derived from configured app name.

	Used for attribution/identity detection without hardcoding product names.
	"""
	app_name = (getattr(settings, "app_name", "") or "").strip().lower()
	if not app_name:
		return []
	parts = [p for p in re.split(r"\W+", app_name) if p]
	targets: set[str] = {app_name}
	# Prefer a non-trivial single-token alias (avoid overly-generic tokens like "ai").
	for p in parts:
		if len(p) >= 4:
			targets.add(p)
			break
	return [t for t in targets if len(t) >= 3]


def developer_name_targets(settings) -> list[str]:
	"""Lowercase identity targets derived from configured developer name."""
	developer = (getattr(settings, "app_developer_name", "") or "").strip().lower()
	if not developer:
		return []
	parts = [p for p in re.split(r"\W+", developer) if p]
	if len(parts) >= 2:
		return [parts[0], parts[-1]]
	return parts


def identity_response_text(settings, question: str) -> str:
	"""Return a deterministic attribution/identity answer.

	We intentionally do NOT call the LLM for these questions to avoid
	provider-specific identity hallucinations (e.g., claiming Google/OpenAI).
	"""
	app_name, developer, attribution = get_app_identity(settings)
	developer = (developer or "").strip()
	attribution = (attribution or "").strip()
	q = (question or "").lower()
	# NOTE: uses deterministic template; do not call LLM here.
	# If the user asks for the assistant/app name, answer directly.
	if re.search(r"\b(what\s*(?:'s| is)\s+your\s+name|whats\s+your\s+name|what\s+is\s+you\s+name)\b", q):
		return f"My name is {app_name}.\n\n{attribution}"

	developer_first = (developer.split()[0] if developer else "").strip()

	def _attribution_fallback() -> str:
		return (
			f"{app_name} is an independently developed platform. For official information about its development or ownership, "
			f"please refer to {app_name}’s documentation or website."
		)

	def _base() -> str:
		# Keep this concise, deterministic, and accurate.
		# First sentence is intentionally product-style for the UI.
		if not developer_first:
			return (
				f"I’m {app_name} — an interview preparation assistant application that helps candidates practice smarter, "
				"get structured feedback, and track progress. "
				"I use AI language models via API providers and add app-specific orchestration to produce interview-ready outputs. "
				"For information about development or ownership, please refer to official documentation."
			)
		return (
			f"I’m {app_name} — an interview preparation assistant application built and maintained by {developer_first} "
			"to help candidates practice smarter, get structured feedback, and track real progress. "
			"I use AI language models via API providers and add app-specific orchestration to produce interview-ready outputs."
		)

	# Founder/creator/owner questions should be deterministic and concise.
	# This branch is intentionally distinct from the generic identity branch so the UI
	# doesn't show identical answers for different user intents.
	if any(w in q for w in ["founder", "creator", "owner", "built", "made", "developed", "develop"]):
		app_targets = app_name_targets(settings)
		# Only treat as product attribution when the user is clearly referring to the app/this application.
		# Do NOT match generic identity questions like "who are you?".
		if any(t in q for t in app_targets) or "this app" in q or "this application" in q:
			if not developer:
				return _attribution_fallback()
			return (
				f"{app_name} is an interview-prep assistant application built and maintained by {developer}. "
				"It helps candidates practice smarter, get structured feedback, and track progress."
			)

		# Also handle direct assistant-addressed phrasing (e.g., "who developed you") without
		# accidentally catching "who are you".
		if "you" in q and ("develop" in q or "created" in q or "built" in q or "made" in q or "founder" in q or "owner" in q):
			if not developer:
				return _attribution_fallback()
			return (
				f"{app_name} is an interview-prep assistant application built and maintained by {developer}. "
				"It helps candidates practice smarter, get structured feedback, and track progress."
			)

	# If the user asks who the configured developer is, keep it scoped to product attribution.
	dev_parts = developer_name_targets(settings)
	if dev_parts and all(p in q for p in dev_parts) and ("who is" in q or "tell me about" in q):
		# Even if a developer name is configured, avoid employment/ownership assertions.
		return (
			f"{developer} is listed in the app configuration for {app_name}. "
			"For official details about development or ownership, please refer to official documentation."
		)

	# If the user asks about ownership/creator, answer succinctly.
	app_targets = app_name_targets(settings)
	if ("owner" in q or "owns" in q or "creator" in q or "developer" in q or "built" in q or "made" in q) and (
		any(t in q for t in app_targets) or "you" in q or "this app" in q or "this application" in q
	):
		if not developer:
			return _attribution_fallback()
		return (
			f"{app_name} is an interview preparation assistant application. "
			f"For development/ownership details, refer to official documentation. "
			f"(Configured attribution: {developer}.)"
		)

	# If the user explicitly asks about ChatGPT/OpenAI, clarify relationship safely.
	if "chatgpt" in q or "openai" in q:
		return (
			f"I’m {app_name}. "
			"ChatGPT is a product from OpenAI. "
			f"{app_name} is a separate interview-prep application that may use AI model APIs (depending on configuration)."
		)

	# If the user mentions Google/Gemini, avoid misattribution while staying accurate.
	if "google" in q or "gemini" in q:
		return (
			f"I’m {app_name}. "
			"I may use different AI model providers depending on your settings (for example, Gemini), "
			f"but {app_name} itself is not an official Google product."
		)

	# Default: generic identity description
	return _base()


def is_identity_question(settings, question: str) -> bool:
	"""Detect 'who made you' / attribution questions.

	These should bypass ambiguity/off-topic routing and respond directly.
	"""
	q = (question or "").strip().lower()
	if not q:
		return False

	# Strip punctuation for more robust matching (handles "who are you!!", "who are you?!", etc.)
	q_clean = re.sub(r"[^\w\s]", " ", q)
	q_clean = " ".join(q_clean.split())

	# Common identity intent phrases (product-name specific patterns are derived dynamically).
	patterns = [
		"who developed you",
		"who created you",
		"who built you",
		"who made you",
		"who is your developer",
		"who is your creator",
		"who is your founder",
		"who is the founder",
		"who made this app",
		"who built this app",
		"who developed this app",
		"who created this app",
		"who owns this app",
		"who are you",
		"what are you",
		"what is your name",
		"what's your name",
		"whats your name",
		"what is you name",
	]
	# Check both original and cleaned versions
	if any(p in q or p in q_clean for p in patterns):
		return True

	# Product-definition style queries should use deterministic identity answers
	# (prevents the definition fallback from injecting rigid bullet labels).
	app_targets = app_name_targets(settings)
	if app_targets and any(t in q_clean for t in app_targets):
		# e.g. "what is stratax", "what is stratax meant to be", "tell me about stratax"
		if re.search(r"\b(what\s+is|meaning\s+of|define|definition\s+of|meant\s+to\s+be|purpose\s+of|about)\b", q_clean):
			return True

	# Regex: handle punctuation/extra whitespace around name questions
	if re.search(r"\b(what\s*(?:'s| is)\s+your\s+name|whats\s+your\s+name|what\s+is\s+you\s+name)\b", q_clean):
		return True

	# Questions about the configured developer by name
	dev_parts = developer_name_targets(settings)
	if dev_parts and all(p in q_clean for p in dev_parts) and any(
		x in q_clean for x in ["who is", "developer", "developed", "created", "built", "founder", "owner"]
	):
		return True

	# Small heuristic: actor words near app name / "you"
	has_actor = any(w in q_clean for w in ["developer", "developed", "created", "built", "made", "founder", "owner"])
	app_targets = app_name_targets(settings)
	has_target = any(w in q_clean for w in ["this app", "this application", "you", "your"]) or any(t in q_clean for t in app_targets)
	return has_actor and has_target


def identity_overrides(settings) -> str:
	"""Overrides for identity/attribution questions."""
	app_name, developer, attribution = get_app_identity(settings)
	developer = (developer or "").strip()
	attribution = (attribution or "").strip()
	return (
		"\n\nIdentity/Attribution Overrides (apply only to identity/ownership questions):\n"
		"- Respond in 1–3 short sentences. No long templates, no headings, no bullet lists.\n"
		f"- Identify the product: '{app_name}' is an application/platform (not a standalone model).\n"
		"- Do NOT make claims about creator employment, ownership, or affiliations unless explicitly provided by official documentation or configuration.\n"
		"- If attribution details are missing or uncertain, say you can’t verify and refer to official documentation.\n"
		"- Be accurate: it may use AI language models via API providers; avoid implying the app itself is a standalone model.\n"
		"- If asked about ChatGPT/OpenAI/Google: clarify those are separate companies/products; do not claim affiliation.\n"
		f"- If needed, include configured attribution verbatim: {attribution or 'Refer to official documentation.'}\n"
	)
