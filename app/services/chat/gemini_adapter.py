from __future__ import annotations

"""Thin adapter for optional Gemini / Google Generative AI dependency.

We import this lazily so the app can start without Gemini installed, and so
we do not emit deprecation warnings unless Gemini is actually used.
"""

from typing import Any, Optional
import warnings


def _import_google_generativeai() -> Optional[Any]:
	"""Import the legacy google.generativeai module (best-effort).

	We suppress its FutureWarning on import to keep startup logs clean.
	"""
	try:
		with warnings.catch_warnings():
			# The package emits a FutureWarning about deprecation at import-time.
			# Filter by message text (more reliable than module filtering).
			warnings.filterwarnings(
				"ignore",
				category=FutureWarning,
				message=r"(?s).*support for the `google\.generativeai` package has ended.*",
			)
			import google.generativeai as _genai  # type: ignore
		return _genai
	except Exception:
		return None


class _LazyGenAIProxy:
	def __init__(self) -> None:
		self._module: Optional[Any] = None

	def _load(self) -> Any:
		if self._module is not None:
			return self._module
		mod = _import_google_generativeai()
		if mod is None:
			# NOTE: google-genai is installed in some environments, but this codebase
			# currently uses the legacy google.generativeai API (configure/GenerativeModel).
			raise ImportError(
				"Gemini dependency not available. Install 'google-generativeai' or update the codebase to 'google-genai'."
			)
		self._module = mod
		return mod

	def __getattr__(self, item: str) -> Any:
		return getattr(self._load(), item)

	def __bool__(self) -> bool:
		# Treat as "available" only if import succeeds.
		try:
			self._load()
			return True
		except Exception:
			return False


# Backwards-compatible export used across the codebase.
genai = _LazyGenAIProxy()
