"""Planning step for architecture generation.

Replaces two brittle keyword heuristics with one cheap structured LLM call.

The problem it solves
---------------------
``_estimate_problem_complexity`` scored the *prompt text*, not the system being
designed. It looked for constraint words ("scale", "multi-region", "low
latency"), so anything phrased plainly scored zero::

    score  tier     query
        0  simple   design netflix
        0  simple   design a url shortener
        0  simple   design a microsoft teams with neat architecture
        6  medium   design a globally scalable multi-region low latency system

Netflix and a URL shortener received an identical 7-node budget, while a prompt
that merely *said* buzzwords got a bigger one. Separately, the chat path
hardcoded 5 views for every request, so a URL shortener cost the same 5-10
sequential LLM calls as Teams — which is what exhausted the Groq TPM budget and
turned one request into 68 seconds of 429 backoff.

The planner asks the model what the system actually is, then decides how many
views it warrants. Fewer views for simple systems means fewer LLM calls, which
is the direct fix for the rate-limit storms.

Safety
------
This runs before generation, so a failure here must never block a diagram. Every
failure path — bad JSON, provider outage, rate limit, nonsense values — falls
back to the deterministic keyword heuristic that shipped before. The planner can
only ever change *how many* views and *how big*, never whether generation runs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# Views the chat path can produce, in the order a reader should meet them.
# Kept as plain strings so this module does not import the router's enum.
ALL_VIEWS = [
    "SYSTEM_OVERVIEW",
    "REQUEST_FLOW",
    "DATA_MODEL",
    "DEPLOYMENT",
    "OBSERVABILITY",
]

# Never drop these — a system design answer without an overview and a request
# path is not an answer.
MANDATORY_VIEWS = ["SYSTEM_OVERVIEW", "REQUEST_FLOW"]

VALID_TIERS = ("simple", "medium", "complex")

# How many views each tier justifies. A URL shortener does not need a
# deployment topology and an observability plan to be a good answer.
TIER_VIEW_BUDGET = {
    "simple": 2,
    "medium": 3,
    "complex": 5,
}


@dataclass
class ArchitecturePlan:
    """Result of the planning step."""

    tier: str
    views: List[str]
    source: str  # "llm" | "heuristic"
    reasoning: str = ""
    signals: List[str] = field(default_factory=list)

    @property
    def is_llm(self) -> bool:
        return self.source == "llm"


_PLANNER_SYSTEM_PROMPT = """You are a staff engineer scoping a system design answer.

Judge the SYSTEM being designed, not how the question is worded. "Design Netflix"
is complex even though it is three words. "Design a globally scalable low-latency
service" is vague and may be simple.

Return STRICT JSON, no prose, no code fences:

{
  "tier": "simple" | "medium" | "complex",
  "reasoning": "<one short sentence>",
  "signals": ["<what makes it this tier>", "..."]
}

Tier guidance:
- simple:  one primary data flow, no fan-out, a single store would do.
           (url shortener, pastebin, todo API, key-value cache)
- medium:  several cooperating services, async work, or a real scale concern.
           (ride hailing for one city, ticket booking, chat for a small team)
- complex: global scale, multi-region, heavy media or fan-out, strict
           consistency or latency budgets, many interacting subsystems.
           (netflix, whatsapp, google calendar, microsoft teams, uber, twitter)

Judge by inherent difficulty. Do not inflate a tier because the prompt uses
words like "scalable" or "production-grade"."""


def _heuristic_plan(system_description: str, settings_obj=None) -> ArchitecturePlan:
    """Deterministic fallback: the original keyword scoring.

    Preserved exactly so behaviour is unchanged whenever the planner is
    unavailable. Known to under-rate plainly worded systems — that is the
    limitation the LLM path exists to cover, not something to fix here.
    """
    if settings_obj is None:
        from app.config import settings as settings_obj  # local import keeps this testable

    text = (system_description or "").lower()
    if not text.strip():
        return ArchitecturePlan(tier="simple", views=list(MANDATORY_VIEWS), source="heuristic")

    signals = list(getattr(settings_obj, "architecture_complexity_signals", []) or [])
    score = sum(1 for kw in signals if kw in text)
    if len(text) > 120:
        score += 1
    if len(text) > 240:
        score += 1

    if score >= 9:
        tier = "complex"
    elif score >= 4:
        tier = "medium"
    else:
        tier = "simple"

    return ArchitecturePlan(
        tier=tier,
        views=views_for_tier(tier),
        source="heuristic",
        reasoning=f"keyword score {score}",
    )


def views_for_tier(tier: str) -> List[str]:
    """Views a tier justifies, always including the mandatory ones."""
    budget = TIER_VIEW_BUDGET.get(tier, TIER_VIEW_BUDGET["medium"])
    return ALL_VIEWS[:max(len(MANDATORY_VIEWS), budget)]


def _coerce_tier(raw) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    tier = raw.strip().lower()
    return tier if tier in VALID_TIERS else None


def _parse_plan(text: str) -> Optional[ArchitecturePlan]:
    """Parse the planner response. Returns None if unusable."""
    if not text or not text.strip():
        return None

    stripped = text.strip()
    brace = stripped.find("{")
    if brace == -1:
        return None

    try:
        # raw_decode tolerates trailing prose after the object, which providers
        # emit even in JSON mode.
        parsed, _ = json.JSONDecoder().raw_decode(stripped[brace:])
    except Exception:
        return None

    if not isinstance(parsed, dict):
        return None

    tier = _coerce_tier(parsed.get("tier"))
    if tier is None:
        return None

    raw_signals = parsed.get("signals")
    signals = [str(s) for s in raw_signals][:5] if isinstance(raw_signals, list) else []

    return ArchitecturePlan(
        tier=tier,
        views=views_for_tier(tier),
        source="llm",
        reasoning=str(parsed.get("reasoning") or "")[:200],
        signals=signals,
    )


async def plan_architecture(
    system_description: str,
    llm_service=None,
    api_key: Optional[str] = None,
) -> ArchitecturePlan:
    """Decide complexity tier and view set for a system design request.

    Never raises. Falls back to the keyword heuristic on any failure so a
    planning problem can never prevent a diagram from being generated.
    """
    description = (system_description or "").strip()
    if not description:
        return _heuristic_plan(description)

    if llm_service is None:
        try:
            from app.services.chat.llm_service import get_llm_service

            llm_service = get_llm_service(feature="default")
        except Exception:
            logger.warning("Architecture planner: no LLM service; using heuristic", exc_info=True)
            return _heuristic_plan(description)

    if not getattr(llm_service, "enabled", False):
        return _heuristic_plan(description)

    try:
        raw = await llm_service.generate_text(
            prompt=f"System to design: {description}",
            system_prompt=_PLANNER_SYSTEM_PROMPT,
            api_key=api_key,
            json_mode=True,
            temperature=0.0,   # a scoping decision should be stable across identical inputs
            max_tokens=200,    # the response is three short fields
        )
    except Exception as e:
        logger.warning("Architecture planner LLM call failed (%s); using heuristic", e)
        return _heuristic_plan(description)

    plan = _parse_plan(raw)
    if plan is None:
        logger.warning("Architecture planner returned unusable output; using heuristic")
        return _heuristic_plan(description)

    logger.info(
        "🧭 Architecture plan: tier=%s views=%d source=llm (%s)",
        plan.tier,
        len(plan.views),
        plan.reasoning or "no reasoning given",
    )
    return plan
