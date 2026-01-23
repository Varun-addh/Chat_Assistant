from __future__ import annotations

from typing import Optional
from app.config import settings


class DynamicBudgetEngine:
    """Compute an adaptive token budget based on intent, depth, question length,
    and user tier. This replaces hardcoded token buckets with a unit-based
    approach that is easy to tune and observe.

    Formula (simple):
      tokens = token_per_budget_unit
               * intent_unit
               * depth_multiplier
               * length_scale
               * user_tier_multiplier

    The result is clamped to configured ceilings when present.
    """

    DEFAULT_INTENT_UNITS = {
        "general": 1.0,
        "system_design": 2.0,
        "coding": 1.5,
        "mirror": 1.2,
        "database_schema": 1.2,
        "ui_design": 1.2,
        "technical_strategy": 1.2,
        "greeting": 0.5,
        "off_topic": 0.5,
    }

    DEFAULT_DEPTH_MULT = {"quick": 0.6, "standard": 1.0, "deep": 1.6}

    DEFAULT_USER_TIERS = {"free": 0.8, "standard": 1.0, "pro": 1.25, "internal": 1.5}

    @classmethod
    def compute_budget_tokens(
        cls,
        *,
        question: str,
        intent: str = "general",
        depth: str = "standard",
        user_tier: str = "standard",
        base_limit: Optional[int] = None,
    ) -> int:
        q_words = max(1, len((question or "").split()))
        # length scale smoothly 1.0..2.0 for 1..100+ words
        length_scale = 1.0 + min(1.0, q_words / 100.0)

        intent_unit = cls.DEFAULT_INTENT_UNITS.get(intent, 1.0)
        depth_mult = cls.DEFAULT_DEPTH_MULT.get(depth, 1.0)
        user_mult = cls.DEFAULT_USER_TIERS.get(user_tier, 1.0)

        token_per_unit = getattr(settings, "token_per_budget_unit", 400)

        raw = int(token_per_unit * intent_unit * depth_mult * length_scale * user_mult)

        # sensible floor and ceiling
        floor = max(150, int(token_per_unit * 0.5))
        # ceiling: use base_limit if provided, else respect groq_max_tokens_complex if set
        ceiling = None
        if base_limit:
            ceiling = int(base_limit)
        elif getattr(settings, "groq_max_tokens_complex", None):
            ceiling = int(getattr(settings, "groq_max_tokens_complex"))

        if ceiling is not None:
            target = max(floor, min(raw, ceiling))
        else:
            target = max(floor, raw)

        return int(target)


# Singleton convenience
dynamic_budget_engine = DynamicBudgetEngine()
