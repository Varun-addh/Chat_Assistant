"""Regression tests for the confidence scoring curve.

Bug (found in production logs, 2026-07-27): every answer in a real session
scored exactly 1.0 confidence.

    'confidence_score': 1.0   pitch_variance 2744.46
    'confidence_score': 1.0   pitch_variance 2694.14
    'confidence_score': 1.0   pitch_variance 2433.05

The sweet-spot branch was::

    confidence_score = 0.9 + min(0.1, (pitch_variance - 800) / 10000)

The min() saturates once (pv - 800) / 10000 >= 0.1, i.e. at pv >= 1800. The
sweet spot spans 800-3500, so 63% of it — and effectively all natural speech —
was pinned to a perfect score. Confidence feeds the delivery dimension, which
is 25% of the overall practice score, so that quarter was a constant.

These tests pin the curve to the range its own docstring claims, and assert it
can actually distinguish two different speakers.
"""

import pytest

from app.services.practice.speech_analytics_agent import SpeechAnalyticsAgent
from app.schemas import SpeechAnalyticsConfig


# The real implementation — not a copy. The curve is a pure function precisely
# so these tests bind to the code that actually runs in production.
_confidence_for = SpeechAnalyticsAgent.confidence_from_pitch_variance


# The exact pitch variances from the production session that exposed the bug.
PRODUCTION_VARIANCES = [2744.4622, 2694.1443, 2433.0496]


def test_curve_is_reachable_from_the_agent_instance():
    """The instance path used by _calculate_confidence resolves to this curve."""
    agent = SpeechAnalyticsAgent(config=SpeechAnalyticsConfig())
    assert agent.confidence_from_pitch_variance(2400) == _confidence_for(2400)


@pytest.mark.parametrize("pitch_variance", PRODUCTION_VARIANCES)
def test_normal_speech_is_not_a_perfect_score(pitch_variance):
    """Ordinary speech must not score a flat 1.0."""
    score = round(_confidence_for(pitch_variance), 2)
    assert score < 1.0, f"pv={pitch_variance} still saturates at {score}"
    assert score > 0.9, f"pv={pitch_variance} should still be a good score, got {score}"


def test_sweet_spot_only_reaches_1_0_at_the_top():
    """Only the very top of the band earns a perfect score."""
    assert round(_confidence_for(800), 2) == 0.9
    assert round(_confidence_for(3500), 2) == 1.0
    # Everything strictly inside the band is strictly between.
    for pv in (1200, 1800, 2400, 3000):
        assert 0.9 < _confidence_for(pv) < 1.0


def test_curve_is_monotonic_across_sweet_spot():
    """Higher variance within the sweet spot should not score lower."""
    scores = [_confidence_for(pv) for pv in range(800, 3501, 100)]
    assert scores == sorted(scores)


def test_curve_discriminates_between_speakers():
    """Two speakers with clearly different variance get different scores.

    This is the property the bug destroyed: with the saturating min(), pv=1800
    and pv=3500 both returned exactly 1.0.
    """
    steady = round(_confidence_for(1800), 2)
    varied = round(_confidence_for(3400), 2)
    assert steady != varied, f"curve cannot distinguish speakers ({steady} == {varied})"


def test_rounding_preserves_distinctions():
    """Stored precision must not collapse the band back to {0.9, 1.0}.

    confidence_score is persisted with round(x, 2); at 1 decimal the entire
    0.9-1.0 sweet spot degenerates to two possible values.
    """
    distinct = {round(_confidence_for(pv), 2) for pv in range(800, 3501, 100)}
    assert len(distinct) > 2, f"only {len(distinct)} distinct values across the band"


@pytest.mark.parametrize(
    "pitch_variance,expected_band",
    [
        (400, "monotone"),
        (2400, "sweet"),
        (5000, "wavering"),
        (9000, "shaky"),
    ],
)
def test_bands_stay_ordered(pitch_variance, expected_band):
    """Band ordering: sweet > monotone > wavering > shaky."""
    score = _confidence_for(pitch_variance)
    if expected_band == "sweet":
        assert score > 0.9
    elif expected_band == "monotone":
        assert 0.7 <= score <= 0.95
    elif expected_band == "wavering":
        assert 0.5 <= score < 0.9
    else:
        assert score < 0.5
