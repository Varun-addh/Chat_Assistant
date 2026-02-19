from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AbuseResult:
    is_abusive: bool
    reason: str


# Conservative patterns: only trigger when directed at the assistant/product/developer.
_INSULT_WORDS = [
    "idiot",
    "stupid",
    "dumb",
    "moron",
    "trash",
    "fraud",
    "scam",
    "scammer",
    "thief",
    "liar",
    "ripoff",
    "con artist",
]

# Common directed profanity/abuse phrases.
_DIRECTED_PHRASES = [
    "fuck you",
    "f**k you",
    "go to hell",
]


def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def classify_abuse(
    text: str,
    *,
    app_name: Optional[str] = None,
    developer_name: Optional[str] = None,
) -> AbuseResult:
    """Heuristic abuse/defamation detector.

    Goal: avoid engaging in abusive/defamatory turns while minimizing false positives
    for normal interview questions.

    We only treat it as abuse if the language is directed at the assistant/product/
    configured developer (2nd-person or named target), not when discussing abuse
    as a topic.
    """

    t = _normalize(text)
    if not t:
        return AbuseResult(is_abusive=False, reason="empty")

    # Quick directed phrase check.
    for p in _DIRECTED_PHRASES:
        if p in t:
            return AbuseResult(is_abusive=True, reason="directed_phrase")

    # Determine whether the message targets the assistant/product/developer.
    targets: list[str] = []
    if app_name:
        targets.append(app_name.strip().lower())
        # Also include a non-trivial token alias from app name.
        tokens = [x for x in re.split(r"\W+", app_name.lower()) if x]
        for tok in tokens:
            if len(tok) >= 4 and tok not in targets:
                targets.append(tok)
                break

    if developer_name:
        parts = [x for x in re.split(r"\W+", developer_name.lower()) if x]
        targets.extend(parts[:1] + parts[-1:])

    # Second-person is a strong signal.
    directed = any(w in t.split() for w in ["you", "your", "u"])
    if not directed and targets:
        directed = any(trg and trg in t for trg in targets)

    if not directed:
        return AbuseResult(is_abusive=False, reason="not_directed")

    # Match insult words/phrases (word-boundary for single tokens).
    for w in _INSULT_WORDS:
        if " " in w:
            if w in t:
                return AbuseResult(is_abusive=True, reason=f"insult:{w}")
        else:
            if re.search(rf"\b{re.escape(w)}\b", t):
                return AbuseResult(is_abusive=True, reason=f"insult:{w}")

    # Money/grievance defamation patterns.
    if re.search(r"\b(took|stole|steal|scammed|cheated)\b", t) and re.search(r"\bmoney\b", t):
        return AbuseResult(is_abusive=True, reason="money_accusation")

    return AbuseResult(is_abusive=False, reason="no_match")
