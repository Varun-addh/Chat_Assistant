from __future__ import annotations

from dataclasses import dataclass

from app.prompts.policies import (
	ACCURACY_AND_CALIBRATION,
	DEPTH_BUDGET,
    ALGO_DIAGRAM,
    BASE_PERSONA,
    BEHAVIORAL_STAR,
    CODE_QUALITY,
    CONCEPT_QUALITY,
    DB_SCHEMA_DIAGRAM,
    DIAGRAMS_GENERAL,
    IDENTITY_AND_ATTRIBUTION,
	OUTPUT_HYGIENE,
	RESPONSE_CONTRACT,
    COPILOT_SYSTEM,
    PROMPT_INJECTION_RESISTANCE,
    COPILOT_INTERVIEW_MODE,
    RESPONSE_TEMPLATE,
    OUTPUT_GUARDS,
    SYSTEM_DESIGN,
    SYSTEM_DESIGN_DIAGRAM,
    UI_DIAGRAM,
	UX_CONVERSATION,
    PolicyModule,
    join_policy_modules,
    render_policy,
)

from app.prompts.mirror_policies import MIRROR_MODE


@dataclass(frozen=True)
class PromptFlags:
    is_system_design: bool = False
    is_database_schema: bool = False
    is_ui_design: bool = False
    is_algorithm: bool = False
    needs_first_person: bool = False
    is_technical_strategy: bool = False
    is_mirror_mode: bool = False

    # quick | standard | deep (best-effort inferred)
    depth: str = "standard"


def build_default_system_prompt(
    *,
    app_name: str,
    developer_name: str,
    attribution: str,
    flags: PromptFlags,
) -> str:
    """Build the default system prompt by composing small policy modules.

    Goal: reduce contradictions and token bloat by including only what each request needs.
    """

    modules: list[PolicyModule] = [
        COPILOT_SYSTEM,
        PROMPT_INJECTION_RESISTANCE,
        COPILOT_INTERVIEW_MODE,
        RESPONSE_TEMPLATE,
        OUTPUT_GUARDS,
        BASE_PERSONA,
        IDENTITY_AND_ATTRIBUTION,
        UX_CONVERSATION,
        OUTPUT_HYGIENE,
        RESPONSE_CONTRACT,
        DEPTH_BUDGET,
        ACCURACY_AND_CALIBRATION,
    ]

    # Mirror mode uses a different contract (JSON analysis), so avoid conflicting UX contracts.
    if flags.is_mirror_mode:
        modules = [
            BASE_PERSONA,
            IDENTITY_AND_ATTRIBUTION,
            OUTPUT_HYGIENE,
            ACCURACY_AND_CALIBRATION,
            MIRROR_MODE,
        ]

    # Make depth explicit so the model can follow the budget deterministically.
    modules.append(PolicyModule(name="depth_selected", text=f"Active depth: {flags.depth}"))

    if not flags.is_mirror_mode:
        # Technical modes
        if flags.is_system_design:
            modules.extend([SYSTEM_DESIGN, DIAGRAMS_GENERAL, SYSTEM_DESIGN_DIAGRAM])
        if flags.is_database_schema:
            modules.extend([DIAGRAMS_GENERAL, DB_SCHEMA_DIAGRAM])
        if flags.is_ui_design:
            modules.extend([DIAGRAMS_GENERAL, UI_DIAGRAM])
        if flags.is_algorithm:
            modules.extend([CODE_QUALITY, DIAGRAMS_GENERAL, ALGO_DIAGRAM])

        # Behavior/persona mode
        if flags.needs_first_person and not flags.is_technical_strategy:
            modules.append(BEHAVIORAL_STAR)

        # Default coverage for normal technical Qs
        if not (flags.is_system_design or flags.is_database_schema or flags.is_ui_design):
            modules.extend([CODE_QUALITY, CONCEPT_QUALITY])

    rendered = []
    for m in modules:
        rendered.append(
            PolicyModule(
                name=m.name,
                text=render_policy(m.text, app_name=app_name, developer_name=developer_name, attribution=attribution),
            )
        )

    return join_policy_modules(rendered)
