from __future__ import annotations

"""Thin adapter for optional Gemini / Google Generative AI dependency.

We import this behind a try/except in multiple places so the app can run
without Gemini installed.
"""

try:
	import google.generativeai as genai  # type: ignore
except Exception:
	genai = None
