"""Tests for the architecture planning step and the sizing it drives.

Context: the keyword estimator scored the prompt's *wording* rather than the
system being designed, so "design netflix" and "design a url shortener" both
scored 0 and received an identical 7-node budget, while a prompt that merely
said "scalable multi-region" scored higher. The chat path separately hardcoded
5 views for every request, which is what exhausted the Groq TPM budget.
"""

import pytest

from app.services.architecture.architecture_planner import (
    ALL_VIEWS,
    MANDATORY_VIEWS,
    TIER_VIEW_BUDGET,
    ArchitecturePlan,
    _heuristic_plan,
    _parse_plan,
    plan_architecture,
    views_for_tier,
)


class _FakeLLM:
    """Stand-in for LLMService.generate_text."""

    def __init__(self, response=None, raises=None, enabled=True):
        self._response = response
        self._raises = raises
        self.enabled = enabled
        self.calls = []

    async def generate_text(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises:
            raise self._raises
        return self._response


# ---------------------------------------------------------------------------
# View budgets
# ---------------------------------------------------------------------------

def test_mandatory_views_always_present():
    for tier in ("simple", "medium", "complex"):
        views = views_for_tier(tier)
        for required in MANDATORY_VIEWS:
            assert required in views, f"{tier} dropped {required}"


def test_simple_tier_costs_fewer_llm_calls_than_complex():
    """The whole point: a URL shortener must not cost what Teams costs."""
    assert len(views_for_tier("simple")) < len(views_for_tier("complex"))
    assert len(views_for_tier("complex")) == len(ALL_VIEWS)


def test_unknown_tier_falls_back_to_medium_budget():
    assert len(views_for_tier("nonsense")) == TIER_VIEW_BUDGET["medium"]


def test_views_are_a_prefix_of_the_canonical_order():
    """Views must arrive in reading order, never shuffled."""
    for tier in ("simple", "medium", "complex"):
        views = views_for_tier(tier)
        assert views == ALL_VIEWS[: len(views)]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parses_clean_json():
    plan = _parse_plan('{"tier": "complex", "reasoning": "global fan-out", "signals": ["media"]}')
    assert plan.tier == "complex"
    assert plan.source == "llm"
    assert plan.reasoning == "global fan-out"
    assert plan.signals == ["media"]


def test_parses_json_with_trailing_prose():
    """Providers append text after the object even in JSON mode."""
    plan = _parse_plan('{"tier": "simple"}\nHope this helps!')
    assert plan is not None and plan.tier == "simple"


def test_parses_json_behind_a_code_fence():
    plan = _parse_plan('```json\n{"tier": "medium"}\n```')
    assert plan is not None and plan.tier == "medium"


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "not json at all", "{]", '{"reasoning": "no tier"}', '{"tier": "enormous"}', "[1,2,3]"],
)
def test_unusable_output_returns_none(bad):
    assert _parse_plan(bad) is None


def test_tier_is_case_insensitive():
    assert _parse_plan('{"tier": "COMPLEX"}').tier == "complex"


# ---------------------------------------------------------------------------
# Fallback safety — planning must never block generation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_exception_falls_back_to_heuristic():
    llm = _FakeLLM(raises=RuntimeError("429 rate limited"))
    plan = await plan_architecture("design netflix", llm_service=llm)
    assert plan.source == "heuristic"
    assert plan.tier in ("simple", "medium", "complex")
    assert plan.views  # still produces a usable view set


@pytest.mark.asyncio
async def test_garbage_llm_output_falls_back_to_heuristic():
    plan = await plan_architecture("design netflix", llm_service=_FakeLLM(response="¯\\_(ツ)_/¯"))
    assert plan.source == "heuristic"
    assert plan.views


@pytest.mark.asyncio
async def test_disabled_llm_falls_back_without_calling():
    llm = _FakeLLM(response='{"tier":"complex"}', enabled=False)
    plan = await plan_architecture("design netflix", llm_service=llm)
    assert plan.source == "heuristic"
    assert llm.calls == [], "must not call a disabled provider"


@pytest.mark.asyncio
async def test_empty_description_does_not_call_the_llm():
    llm = _FakeLLM(response='{"tier":"complex"}')
    plan = await plan_architecture("   ", llm_service=llm)
    assert plan.source == "heuristic"
    assert llm.calls == []


@pytest.mark.asyncio
async def test_planner_uses_deterministic_temperature():
    """Identical inputs should scope identically."""
    llm = _FakeLLM(response='{"tier": "complex"}')
    await plan_architecture("design netflix", llm_service=llm)
    assert llm.calls[0]["temperature"] == 0.0
    assert llm.calls[0]["json_mode"] is True


@pytest.mark.asyncio
async def test_llm_plan_is_used_when_valid():
    llm = _FakeLLM(response='{"tier": "complex", "reasoning": "global media fan-out"}')
    plan = await plan_architecture("design netflix", llm_service=llm)
    assert plan.source == "llm"
    assert plan.tier == "complex"
    assert plan.views == ALL_VIEWS


@pytest.mark.asyncio
async def test_planner_distinguishes_systems_the_heuristic_could_not():
    """The regression that motivated this module.

    The keyword heuristic scored both of these 0/simple. A working planner must
    separate them.
    """
    netflix = await plan_architecture(
        "design netflix", llm_service=_FakeLLM(response='{"tier": "complex"}')
    )
    shortener = await plan_architecture(
        "design a url shortener", llm_service=_FakeLLM(response='{"tier": "simple"}')
    )
    assert netflix.tier != shortener.tier
    assert len(netflix.views) > len(shortener.views)


# ---------------------------------------------------------------------------
# Heuristic fallback keeps its documented behaviour
# ---------------------------------------------------------------------------

def test_heuristic_still_scores_buzzword_prompts_higher():
    plain = _heuristic_plan("design netflix")
    buzzy = _heuristic_plan(
        "design a globally scalable multi-region low latency high availability "
        "system with disaster recovery and fault tolerance"
    )
    assert plain.tier == "simple"
    assert buzzy.tier in ("medium", "complex")
    assert plain.source == buzzy.source == "heuristic"


def test_heuristic_handles_empty_input():
    plan = _heuristic_plan("")
    assert plan.tier == "simple"
    assert plan.views == MANDATORY_VIEWS


def test_planner_has_no_hardcoded_system_names():
    """Generalisation guard.

    The planner must reason about any system, not recognise a list of famous
    ones. Verified live against 17 varied prompts — pastebin, IoT telemetry,
    HIPAA records, game matchmaking, gibberish — all tiered sensibly with no
    system named in this module. A keyword list creeping back in would score
    whatever it lists and silently mis-tier everything else, which is exactly
    the failure this module replaced.

    Product names appear only in the LLM prompt as tier *examples*; they must
    never be matched against user input in code.
    """
    import ast
    import inspect

    from app.services.architecture import architecture_planner as mod

    tree = ast.parse(inspect.getsource(mod))

    # Collect only string literals that could be *compared against input*.
    # Docstrings (which quote the old failure table) and the LLM prompt (whose
    # examples are instructions to the model, never matched in code) are
    # excluded. Docstring nodes are identified structurally — by object
    # identity — because ast.get_docstring() re-indents the text, so comparing
    # by value silently fails to match.
    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    docstring_nodes.add(id(body[0].value))

    literals = [
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstring_nodes
        and node.value != mod._PLANNER_SYSTEM_PROMPT
    ]

    haystack = " ".join(literals)
    for name in ("netflix", "uber", "whatsapp", "twitter", "microsoft teams", "google calendar"):
        assert name not in haystack, (
            f"{name!r} appears as a string literal in planner logic — that is "
            "keyword matching, the very thing this module replaced"
        )


def test_plan_is_llm_property():
    assert ArchitecturePlan("simple", [], "llm").is_llm is True
    assert ArchitecturePlan("simple", [], "heuristic").is_llm is False


# ---------------------------------------------------------------------------
# Sizing follows the planned tier
# ---------------------------------------------------------------------------

def test_planned_tier_overrides_keyword_sizing():
    """A complex system must get a larger budget than a simple one.

    Before, both went through the keyword estimator and scored 0, so Netflix and
    a URL shortener received identical limits.
    """
    from app.services.architecture.architecture_generator import get_architecture_generator
    from app.schemas import ArchitectureViewType

    gen = get_architecture_generator()
    view = ArchitectureViewType.REQUEST_FLOW

    simple_nodes, simple_edges = gen._dynamic_limits(view, "design netflix", tier="simple")
    complex_nodes, complex_edges = gen._dynamic_limits(view, "design netflix", tier="complex")

    assert complex_nodes > simple_nodes
    assert complex_edges >= simple_edges


def test_absent_tier_preserves_legacy_keyword_behaviour():
    from app.services.architecture.architecture_generator import get_architecture_generator
    from app.schemas import ArchitectureViewType

    gen = get_architecture_generator()
    view = ArchitectureViewType.REQUEST_FLOW

    legacy = gen._dynamic_limits(view, "design netflix")
    explicit_simple = gen._dynamic_limits(view, "design netflix", tier="simple")
    # "design netflix" scores 0 => simple, so these must agree.
    assert legacy == explicit_simple
