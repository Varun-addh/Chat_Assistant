import pytest
from app.prompts.builder import PromptFlags, build_default_system_prompt


@pytest.mark.fast
def test_builder_renders_identity_and_ux_modules():
    out = build_default_system_prompt(
        app_name="Stratax AI",
        developer_name="Varun Bikkumalla",
        attribution="Stratax AI was developed by Varun Bikkumalla.",
        flags=PromptFlags(),
    )

    # Identity text is rendered (placeholders replaced)
    assert "You are Stratax AI" in out
    assert "developed by Varun Bikkumalla" in out

    # UX and hygiene rules present
    assert "Conversation-first UX" in out
    assert "Output hygiene" in out


@pytest.mark.fast
def test_builder_includes_system_design_diagram_policy_when_flagged():
    out = build_default_system_prompt(
        app_name="Stratax AI",
        developer_name="Varun Bikkumalla",
        attribution="Stratax AI was developed by Varun Bikkumalla.",
        flags=PromptFlags(is_system_design=True),
    )

    assert "System design answers" in out
    assert "Mermaid" in out
    # Updated: diagrams are now on-demand only
    assert "ONLY IF" in out or "explicitly asks" in out


@pytest.mark.fast
def test_builder_includes_db_schema_policy_when_flagged_only():
    out = build_default_system_prompt(
        app_name="Stratax AI",
        developer_name="Varun Bikkumalla",
        attribution="Stratax AI was developed by Varun Bikkumalla.",
        flags=PromptFlags(is_database_schema=True),
    )

    assert "erDiagram" in out
    # Should not include the system-design specific module
    assert "System design answers" not in out
