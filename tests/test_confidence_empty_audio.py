"""Regression test: empty/silent audio must NOT get high confidence.

Bug: When a user submits an answer without speaking, the pitch-variance
analysis runs on background noise which has very low variance, causing
the confidence model to return ~0.95 (= "very confident").  The user
sees "100% confidence" for literally saying nothing.

Fix: _calculate_confidence() now guards on transcript word count and
audio RMS energy.  Empty speech → confidence = 0.0.
"""

import numpy as np
from app.services.practice.speech_analytics_agent import SpeechAnalyticsAgent
from app.schemas import SpeechAnalyticsConfig


def _make_agent() -> SpeechAnalyticsAgent:
    return SpeechAnalyticsAgent(config=SpeechAnalyticsConfig())


class TestEmptyAudioConfidence:
    """Confidence must be 0 when the user says nothing."""

    def test_empty_transcript_returns_zero_confidence(self):
        agent = _make_agent()
        # Simulate 5 seconds of near-silent audio (mic noise)
        sr = 16000
        audio = np.random.randn(sr * 5).astype(np.float32) * 0.001  # very quiet
        conf, pv = agent._calculate_confidence(audio, sr, transcript="")
        assert conf == 0.0, f"Empty transcript should give 0 confidence, got {conf}"

    def test_one_word_transcript_returns_zero_confidence(self):
        agent = _make_agent()
        sr = 16000
        audio = np.random.randn(sr * 3).astype(np.float32) * 0.01
        conf, pv = agent._calculate_confidence(audio, sr, transcript="um")
        assert conf == 0.0, f"Single-word transcript should give 0 confidence, got {conf}"

    def test_two_word_transcript_returns_zero_confidence(self):
        agent = _make_agent()
        sr = 16000
        audio = np.random.randn(sr * 3).astype(np.float32) * 0.01
        conf, pv = agent._calculate_confidence(audio, sr, transcript="I think")
        assert conf == 0.0, f"Two-word transcript should give 0 confidence, got {conf}"

    def test_silent_audio_returns_zero_confidence(self):
        """Completely silent audio (all zeros) → 0 confidence."""
        agent = _make_agent()
        sr = 16000
        audio = np.zeros(sr * 5, dtype=np.float32)
        conf, pv = agent._calculate_confidence(audio, sr, transcript="I explained the concept thoroughly")
        assert conf == 0.0, f"Silent audio should give 0 confidence, got {conf}"

    def test_real_speech_still_gets_confidence(self):
        """A proper transcript with audible audio should get > 0 confidence."""
        agent = _make_agent()
        sr = 16000
        # Simulate speech-like audio with a clear tone (150 Hz)
        t = np.linspace(0, 3, sr * 3, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 150 * t)  # Clear 150Hz tone, audible RMS
        transcript = "REST is an architectural style for designing networked applications"
        conf, pv = agent._calculate_confidence(audio, sr, transcript=transcript)
        assert conf > 0.0, f"Real speech should get positive confidence, got {conf}"
