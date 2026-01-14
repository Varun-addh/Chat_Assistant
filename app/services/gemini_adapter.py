"""Gemini SDK compatibility layer.

The codebase historically used `google.generativeai` (deprecated). Newer
implementations should use `google-genai` (`from google import genai`).

To minimize churn, we expose a small surface area compatible with the old SDK:
- `genai.configure(api_key=...)`
- `genai.GenerativeModel(model_name).generate_content(prompt, generation_config=...)`

If `google-genai` is installed, we use it and avoid importing the deprecated
package (which emits warnings). If not installed, we fall back to the legacy
SDK as a last resort.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Dict

import logging

logger = logging.getLogger(__name__)


def _normalize_model_name(name: str) -> str:
	name = (name or "").strip()
	# Old code sometimes passes "models/<name>". google-genai examples typically
	# use "gemini-1.5-flash" style names.
	if name.startswith("models/"):
		return name[len("models/"):]
	return name


@dataclass
class _CompatResponse:
	"""Normalize response shape to look like google.generativeai responses."""
	_raw: Any

	@property
	def text(self) -> str:
		# google-genai returns `response.text`.
		val = getattr(self._raw, "text", None)
		if isinstance(val, str):
			return val
		# Fallback for unexpected shapes.
		try:
			return str(self._raw)
		except Exception:
			return ""

	@property
	def parts(self) -> list[Any]:
		# Some call sites check `response.parts`.
		parts = getattr(self._raw, "parts", None)
		if isinstance(parts, list):
			return parts
		# If using google-genai, `candidates` may exist.
		cands = getattr(self._raw, "candidates", None)
		if cands:
			return list(cands)
		return []


class _GenAIModelCompat:
	def __init__(self, backend: str, client: Any, model_name: str, **kwargs: Any) -> None:
		self._backend = backend
		self._client = client
		self._model_name = _normalize_model_name(model_name)
		self._kwargs = kwargs

	def generate_content(self, prompt: str, generation_config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> _CompatResponse:
		if self._backend == "google-genai":
			# google-genai: client.models.generate_content(model=..., contents=..., config=...)
			config = generation_config or kwargs.get("config")
			raw = self._client.models.generate_content(
				model=self._model_name,
				contents=prompt,
				config=config,
			)
			return _CompatResponse(raw)

		# legacy google.generativeai: model.generate_content(prompt, generation_config=...)
		raw = self._client.generate_content(prompt, generation_config=generation_config, **kwargs)
		return _CompatResponse(raw)


class _GenAIModuleCompat:
	"""Module-like object compatible with `google.generativeai as genai`."""

	def __init__(self) -> None:
		self._backend: Optional[str] = None
		self._client: Any = None

	def configure(self, api_key: str) -> None:
		api_key = (api_key or "").strip()
		if not api_key:
			raise ValueError("Missing Gemini API key")

		# Prefer new SDK to avoid deprecation warnings.
		try:
			from google import genai as _new_genai  # type: ignore
			self._client = _new_genai.Client(api_key=api_key)
			self._backend = "google-genai"
			return
		except Exception:
			pass

		# Fallback to legacy SDK.
		try:
			import google.generativeai as _legacy_genai  # type: ignore
			_legacy_genai.configure(api_key=api_key)
			self._client = _legacy_genai
			self._backend = "google-generativeai"
			return
		except Exception as e:
			raise ImportError(
				"Gemini SDK not available. Install `google-genai` (preferred) or `google-generativeai`."
			) from e

	def GenerativeModel(self, model_name: str, *args: Any, **kwargs: Any) -> _GenAIModelCompat:
		if not self._client or not self._backend:
			raise RuntimeError("Gemini client not configured. Call genai.configure(api_key=...) first.")

		if self._backend == "google-genai":
			# google-genai doesn't construct model objects; we wrap the client.
			return _GenAIModelCompat(self._backend, self._client, model_name, **kwargs)

		# Legacy path: the module exposes GenerativeModel.
		model = self._client.GenerativeModel(_normalize_model_name(model_name), *args, **kwargs)
		return _GenAIModelCompat(self._backend, model, model_name, **kwargs)


# Exported compat object.
genai = _GenAIModuleCompat()
