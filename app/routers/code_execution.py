from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request

from app.config import settings
from app.database import get_db_context
from app.middleware.auth import get_user_id_from_request
from app.schemas import CodeExecutionIn, CodeExecutionOut, CodeExecutionTestResult, CodeExecutionTraceEvent
from app.services.chat.ai_native_enhancements import CodeExecutionSandbox
from app.utils.code_line_explain import explain_lines
from app.utils.event_logging import track_event

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/code", tags=["code-execution"])


def _normalize_language(lang: str) -> str:
    return (lang or "").strip().lower()


@router.post("/execute", response_model=CodeExecutionOut)
async def execute_code(payload: CodeExecutionIn, http_request: Request) -> CodeExecutionOut:
    """Execute code on the backend using an external sandbox (Judge0/Piston).

    Security posture:
    - Enforced max code/stdin lengths via schema.
    - Hard cap on number of test cases inside the sandbox helper.
    - Never stores raw code unless explicitly requested.

    Notes:
    - Uses Judge0 when `JUDGE0_API_KEY` is configured, otherwise falls back to Piston.
    - If `enable_code_execution` is false, returns 503.
    """

    if not bool(getattr(settings, "enable_code_execution", False)):
        raise HTTPException(status_code=503, detail="Code execution disabled")

    user_id = get_user_id_from_request(http_request) or "guest_unknown"

    language = _normalize_language(payload.language)
    if not language:
        raise HTTPException(status_code=400, detail="Language is required")

    sandbox = CodeExecutionSandbox(
        judge0_api_key=getattr(settings, "judge0_api_key", None),
        judge0_rapidapi_host=getattr(settings, "judge0_rapidapi_host", "judge0-ce.p.rapidapi.com"),
    )

    try:
        raw_test_cases: Optional[list[dict[str, Any]]] = None
        if payload.test_cases:
            raw_test_cases = [
                {"input": tc.input, "expected_output": tc.expected_output}
                for tc in payload.test_cases
            ]

        result = await sandbox.execute_code(
            code=payload.code,
            language=language,
            test_cases=raw_test_cases,
            stdin=payload.stdin or "",
            trace=bool(payload.trace),
            trace_max_events=int(payload.trace_max_events or 2000),
        )

        stdout = str(result.get("output") or "")
        stderr = str(result.get("error") or "")
        status = result.get("status")
        time_s = result.get("execution_time")
        mem_kb = result.get("memory_used")

        test_results = None
        if isinstance(result.get("test_results"), list):
            test_results = []
            for tr in result.get("test_results"):
                try:
                    test_results.append(
                        CodeExecutionTestResult(
                            input=str(tr.get("input") or ""),
                            expected_output=(str(tr.get("expected_output")) if tr.get("expected_output") is not None else None),
                            actual_output=(str(tr.get("actual_output")) if tr.get("actual_output") is not None else None),
                            passed=bool(tr.get("passed")),
                            error=(str(tr.get("error")) if tr.get("error") else None),
                        )
                    )
                except Exception:
                    continue

        # Telemetry (privacy-safe by default)
        try:
            with get_db_context() as db:
                extra: dict[str, Any] = {
                    "language": language,
                    "success": bool(result.get("success")),
                    "status": status,
                    "time_seconds": time_s,
                    "memory_kb": mem_kb,
                    "code_len": len(payload.code or ""),
                    "stdin_len": len(payload.stdin or ""),
                    "test_cases": len(payload.test_cases or []) if payload.test_cases else 0,
                    "trace": bool(payload.trace),
                    "trace_max_events": int(payload.trace_max_events or 2000),
                }
                if bool(payload.store_code) and bool(getattr(settings, "analytics_store_raw_text", False)):
                    extra["code_preview"] = (payload.code or "")[: min(2000, len(payload.code or ""))]
                track_event(
                    db,
                    user_id=user_id,
                    session_id=None,
                    event_type="code_execution",
                    question_text=None,
                    extra=extra,
                )
        except Exception:
            pass

        trace_events = None
        line_explanations = None
        if isinstance(result.get("trace_events"), list):
            trace_events = []

            if bool(payload.trace) and bool(payload.explain_trace):
                try:
                    line_nums = [int(ev.get("line") or 0) for ev in result.get("trace_events") if isinstance(ev, dict)]
                    line_explanations = explain_lines(
                        code=payload.code,
                        language=language,
                        line_numbers=line_nums,
                        max_lines=int(payload.explain_max_lines or 200),
                    )
                except Exception:
                    line_explanations = None

            for ev in result.get("trace_events"):
                if not isinstance(ev, dict):
                    continue
                try:
                    ln = int(ev.get("line") or 0)
                    trace_events.append(
                        CodeExecutionTraceEvent(
                            step=int(ev.get("step") or 0),
                            line=ln,
                            event=str(ev.get("event") or "line"),
                            locals=(ev.get("locals") if isinstance(ev.get("locals"), dict) else None),
                            explanation=(line_explanations.get(ln) if isinstance(line_explanations, dict) else None),
                        )
                    )
                except Exception:
                    continue

        return CodeExecutionOut(
            success=bool(result.get("success")),
            status=str(status) if status is not None else None,
            stdout=stdout,
            stderr=stderr,
            time_seconds=(float(time_s) if time_s is not None else None),
            memory_kb=(int(mem_kb) if mem_kb is not None else None),
            test_results=test_results,
            trace_events=trace_events,
            line_explanations=line_explanations,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Code execution failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
