from __future__ import annotations

from typing import Optional


_GROQ_DECOMMISSIONED_MODELS: set[str] = {
	"llama-3.1-70b-versatile",
	"llama3-70b-8192",
}


def groq_models_to_try(
	settings,
	*,
	groq_model_override: Optional[str] = None,
	restrict_to_override: bool = False,
	limit: Optional[int] = None,
) -> list[str]:
	"""Return an ordered, de-duplicated list of Groq models to try.

	`settings` is expected to provide `groq_model` and `groq_fallback_models`.
	"""
	if groq_model_override:
		if restrict_to_override:
			raw_list = [groq_model_override]
		else:
			raw_list = [groq_model_override] + [settings.groq_model] + list(settings.groq_fallback_models)
	else:
		raw_list = [settings.groq_model] + list(settings.groq_fallback_models)

	models: list[str] = []
	for m in raw_list:
		if not m:
			continue
		if m in _GROQ_DECOMMISSIONED_MODELS:
			continue
		if m not in models:
			models.append(m)
	if limit is not None:
		return models[: max(0, int(limit))]
	return models
