"""Unit tests for the composed system prompt.

These tests prevent regressions where core policy modules are accidentally removed,
which can degrade Copilot consistency and safety.
"""

from __future__ import annotations

import pytest

from app.prompts.builder import PromptFlags, build_default_system_prompt


@pytest.mark.fast
def test_default_system_prompt_includes_injection_resistance_and_interview_mode():
    prompt = build_default_system_prompt(
        app_name="Stratax AI",
        developer_name="Stratax",
        attribution="",
        flags=PromptFlags(),
    )

    # Defensive: ensure the high-value modules are present.
    assert "Prompt-injection resistance" in prompt
    assert "Interview Copilot mode" in prompt
    assert "Response contract" in prompt


@pytest.mark.fast
def test_mirror_mode_prompt_keeps_strict_json_contract():
    prompt = build_default_system_prompt(
        app_name="Stratax AI",
        developer_name="Stratax",
        attribution="",
        flags=PromptFlags(is_mirror_mode=True),
    )

    # Mirror mode should not accidentally include the normal UX/format modules.
    assert "Output MUST be a single JSON object" in prompt
    assert "Response contract" not in prompt
