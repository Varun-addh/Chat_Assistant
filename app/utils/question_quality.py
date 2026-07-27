"""Quality gate for generated interview questions.

Generated questions are cached in Qdrant so later searches are fast. Without a
gate, a bad generation is written once and then served indefinitely — fixing
the generation prompt does not evict what is already cached. That is exactly
what happened in June: the prompt produced questions like

    "Explain the role of HTML in Agentic AI applications"
    "How does Agentic AI use Python for natural language processing"

which cross-product the topic with unrelated technologies. They scored 0.0,
persisted in the cache, and kept being served after the prompt was fixed.

This module is the single definition of "formulaic" shared by two callers:

* the write gate in ``interview_intelligence_service._store_in_vector_db``,
  which keeps new junk out of the cache, and
* ``scripts/clear_question_cache.py --formulaic``, which evicts junk already
  written.

Keeping one definition means a pattern added for cleanup automatically starts
blocking writes too.
"""

from __future__ import annotations

import re
from typing import List, Optional

# Shapes that read as generated filler rather than something an interviewer
# would ask. Each pattern targets a template, not a topic, so they stay valid
# as the question corpus changes.
FORMULAIC_PATTERNS = [
    # "...uses X for/to/in ..." / "...leverages X to..."
    re.compile(r"\b(uses?|leverages?|utili[sz]es?|employs?)\s+[\w.+#]+\s+(for|to|in)\b", re.I),
    # "...is used in/for/to/by X"
    re.compile(r"\b(is|are)\s+used\s+(in|for|to|by)\s+[\w.+#]+", re.I),
    # "...role of <NamedTechnology> in ..."
    #
    # Case-SENSITIVE by design. The failure mode is dragging a named,
    # unrelated technology into an unrelated topic:
    #
    #   "Explain the role of HTML in Agentic AI applications"   <- reject
    #   "Discuss the role of explainability in Agentic AI"      <- keep
    #
    # Both share the template, so the template alone cannot separate them.
    # Named technologies are capitalised (HTML, Python, React, .NET); the
    # concepts a real interviewer probes are lowercase common nouns
    # (explainability, scalability, caching). Matching only the capitalised
    # form keeps the conceptual questions, which a case-insensitive pattern
    # rejected in production.
    #
    # Known cost: "the role of Redis in a caching layer" is a fair question
    # and still gets blocked. That is the accepted side of the trade-off —
    # this gate only bars a question from the CACHE, never from the response,
    # so a false positive costs one regeneration while a false negative
    # serves junk indefinitely.
    re.compile(r"\brole of\s+[A-Z][\w.+#]*(\s+\w+){0,2}\s+in\b"),
    # "How does X enable/use/leverage/power/drive ..."
    re.compile(r"\bhow does\s+[\w.+#]+\s+(enable|use|leverage|power|drive)\b", re.I),
]

# Below this, a question carries no scenario, constraint, or follow-up surface.
MIN_QUESTION_CHARS = 15


def is_formulaic(question: Optional[str]) -> bool:
    """True if the question matches a known filler template."""
    return any(p.search(question or "") for p in FORMULAIC_PATTERNS)


def is_low_quality(question: Optional[str]) -> bool:
    """True if the question should not be cached or served.

    Deliberately narrow: it rejects empty/stub text and known filler templates,
    and nothing else. A gate that guesses at "depth" would silently discard
    legitimate short questions, which is worse than letting a few through.
    """
    text = (question or "").strip()
    if len(text) < MIN_QUESTION_CHARS:
        return True
    return is_formulaic(text)


def rejection_reason(question: Optional[str]) -> Optional[str]:
    """Why a question fails the gate, or None if it passes. For logging."""
    text = (question or "").strip()
    if not text:
        return "empty"
    if len(text) < MIN_QUESTION_CHARS:
        return f"too_short ({len(text)} chars)"
    if is_formulaic(text):
        return "formulaic_template"
    return None


def filter_low_quality(questions: List, text_of=lambda q: getattr(q, "question", None)):
    """Split questions into ``(kept, rejected)``.

    ``text_of`` extracts the question text, so this works for both model objects
    and plain dicts.
    """
    kept, rejected = [], []
    for q in questions:
        (rejected if is_low_quality(text_of(q)) else kept).append(q)
    return kept, rejected
