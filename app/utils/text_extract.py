from __future__ import annotations

from io import BytesIO


def extract_text_from_pdf(data: bytes) -> str:
	"""Extract text from PDF bytes.

	Strategy:
	1) Try PyPDF2 (fast, works on many text PDFs)
	2) Fallback to pdfminer.six (more robust)
	Returns empty string on failure.
	"""
	# Try PyPDF2 first
	try:
		from PyPDF2 import PdfReader  # type: ignore
		reader = PdfReader(BytesIO(data))
		parts: list[str] = []
		for page in reader.pages:
			try:
				text = page.extract_text() or ""
				if text:
					parts.append(text)
			except Exception:
				continue
		if parts:
			return "\n".join(parts)
	except Exception:
		pass

	# Fallback to pdfminer
	try:
		from pdfminer.high_level import extract_text  # type: ignore
		return extract_text(BytesIO(data)) or ""
	except Exception:
		return ""


