"""LangGraph-backed orchestration for Practice Mode.

This is intentionally an OPTIONAL layer:
- Default behavior remains `PracticeModeService` method calls.
- When enabled via `settings.enable_practice_mode_langgraph`, we run selected
  flows (start/submit/ack) through a LangGraph state machine.

If `langgraph` is not installed, this module can still be imported; graph
construction will fail gracefully and callers should fall back.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, TypedDict, cast

from app.schemas import AnswerSubmission

logger = logging.getLogger(__name__)


class _SubmitAnswerState(TypedDict, total=False):
    session_id: str
    question_id: int
    audio_file_path: str
    api_key: Optional[str]

    # Derived
    session: Any
    question: Any

    transcript: str
    stt_metadata: dict
    metrics: Any
    micro_feedback: Any

    is_complete: bool
    response: dict


class PracticeModeGraph:
    """Thin wrapper around LangGraph graphs for Practice Mode."""

    def __init__(self, practice_service: Any):
        self._svc = practice_service
        self._submit_answer_graph = None
        self._available = False

        try:
            # Import lazily so LangGraph remains an optional dependency.
            from langgraph.graph import END, StateGraph  # type: ignore

            self._END = END
            self._StateGraph = StateGraph
            self._available = True
        except Exception as e:
            logger.info(f"LangGraph not available; PracticeModeGraph disabled: {e}")
            self._available = False
            return

        try:
            self._submit_answer_graph = self._build_submit_answer_graph()
        except Exception as e:
            logger.warning(f"Failed to build Practice Mode LangGraph; disabling: {e}")
            self._available = False
            self._submit_answer_graph = None

    @property
    def available(self) -> bool:
        return bool(self._available and self._submit_answer_graph is not None)

    async def start_interview(
        self,
        *,
        difficulty: Any,
        user_profile: Any = None,
        question_count: int = 5,
        round_type: Any = None,
        api_key: Optional[str] = None,
    ):
        # Start flow is complex and already correct; keep it as a single call.
        return await self._svc.start_interview(
            difficulty=difficulty,
            user_profile=user_profile,
            question_count=question_count,
            round_type=round_type,
            api_key=api_key,
        )

    async def submit_answer(
        self,
        *,
        session_id: str,
        question_id: int,
        audio_file_path: str,
        api_key: Optional[str] = None,
    ) -> dict:
        if not self.available:
            return await self._svc.submit_answer(
                session_id=session_id,
                question_id=question_id,
                audio_file_path=audio_file_path,
                api_key=api_key,
            )

        state: _SubmitAnswerState = {
            "session_id": session_id,
            "question_id": int(question_id),
            "audio_file_path": audio_file_path,
            "api_key": api_key,
        }

        compiled = self._submit_answer_graph
        if compiled is None:
            return await self._svc.submit_answer(
                session_id=session_id,
                question_id=question_id,
                audio_file_path=audio_file_path,
                api_key=api_key,
            )

        # LangGraph API differs slightly across versions. Prefer ainvoke.
        if hasattr(compiled, "ainvoke"):
            out_state = await compiled.ainvoke(state)  # type: ignore[attr-defined]
        else:
            loop = asyncio.get_event_loop()
            out_state = await loop.run_in_executor(None, lambda: compiled.invoke(state))  # type: ignore[attr-defined]

        out_state = cast(dict, out_state)
        return cast(dict, out_state.get("response") or {})

    async def get_next_question_after_acknowledgment(
        self,
        *,
        session_id: str,
        question_id: int,
        api_key: Optional[str] = None,
    ) -> dict:
        # The ack flow includes adaptive follow-up injection + TTS; keep it as a single call.
        return await self._svc.get_next_question_after_acknowledgment(
            session_id=session_id,
            question_id=question_id,
            api_key=api_key,
        )

    # ---- Graph building (Submit Answer) ----

    def _build_submit_answer_graph(self):
        StateGraph = self._StateGraph
        END = self._END

        graph = StateGraph(_SubmitAnswerState)
        graph.add_node("load_session", self._n_load_session)
        graph.add_node("transcribe", self._n_transcribe)
        graph.add_node("analyze", self._n_analyze)
        graph.add_node("micro_feedback", self._n_micro_feedback)
        graph.add_node("store_answer", self._n_store_answer)
        graph.add_node("maybe_evaluate", self._n_maybe_evaluate)
        graph.add_node("build_response", self._n_build_response)

        graph.set_entry_point("load_session")
        graph.add_edge("load_session", "transcribe")
        graph.add_edge("transcribe", "analyze")
        graph.add_edge("analyze", "micro_feedback")
        graph.add_edge("micro_feedback", "store_answer")
        graph.add_edge("store_answer", "maybe_evaluate")
        graph.add_edge("maybe_evaluate", "build_response")
        graph.add_edge("build_response", END)

        return graph.compile()

    async def _n_load_session(self, state: _SubmitAnswerState) -> dict:
        session_id = str(state.get("session_id") or "")
        question_id = int(state.get("question_id") or 0)

        session = getattr(self._svc, "sessions", {}).get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Mirror PracticeModeService behavior (best-effort).
        try:
            from app.utils.time import utcnow

            session.last_activity_at = utcnow()
        except Exception:
            pass

        questions = getattr(session, "questions", None) or []
        if question_id < 1 or question_id > len(questions):
            raise ValueError(f"Invalid question_id {question_id}")

        question = questions[question_id - 1]
        return {"session": session, "question": question}

    async def _n_transcribe(self, state: _SubmitAnswerState) -> dict:
        audio_file_path = str(state.get("audio_file_path") or "")
        transcript, stt_metadata = await self._svc.stt_service.transcribe_async(audio_file_path)
        logger.info(f"Transcription (full): '{transcript}'")
        return {"transcript": transcript, "stt_metadata": stt_metadata}

    async def _n_analyze(self, state: _SubmitAnswerState) -> dict:
        audio_file_path = str(state.get("audio_file_path") or "")
        question = state["question"]
        transcript = str(state.get("transcript") or "")
        stt_metadata = cast(dict, state.get("stt_metadata") or {})

        metrics = self._svc.analytics_agent.analyze_audio(
            audio_path=audio_file_path,
            transcript=transcript,
            time_limit=getattr(question, "time_limit", 90),
            stt_metadata=stt_metadata,
        )
        return {"metrics": metrics}

    async def _n_micro_feedback(self, state: _SubmitAnswerState) -> dict:
        question = state["question"]
        transcript = str(state.get("transcript") or "")
        metrics = state["metrics"]

        micro_feedback = await self._svc.adaptive_interviewer.generate_micro_feedback(
            metrics,
            question_text=getattr(question, "text", ""),
            transcript=transcript,
            question_key_points=getattr(question, "key_points", None),
            question_expected_answer=getattr(question, "expected_answer_template", None),
            question_category=getattr(question, "category", None),
            api_key=state.get("api_key"),
        )
        return {"micro_feedback": micro_feedback}

    async def _n_store_answer(self, state: _SubmitAnswerState) -> dict:
        session = state["session"]
        question_id = int(state.get("question_id") or 0)
        transcript = str(state.get("transcript") or "")
        metrics = state["metrics"]
        micro_feedback = state["micro_feedback"]

        answer = AnswerSubmission(
            question_id=question_id,
            transcript=transcript,
            metrics=metrics,
            micro_feedback=micro_feedback,
            audio_duration=getattr(metrics, "duration", 0.0),
        )

        # Ensure session.answers exists and is the actual stored list.
        if not hasattr(session, "answers") or getattr(session, "answers") is None:
            setattr(session, "answers", [])
        session.answers.append(answer)
        session.current_question_index = question_id - 1

        is_complete = self._svc.interviewer_agent.is_interview_complete(
            session.current_question_index,
            len(getattr(session, "questions", []) or []),
        )

        if not is_complete:
            session.pending_next_question = True
        else:
            session.is_complete = True
            try:
                from app.utils.time import utcnow

                session.completed_at = utcnow()
            except Exception:
                session.completed_at = None

        return {"is_complete": bool(is_complete)}

    async def _n_maybe_evaluate(self, state: _SubmitAnswerState) -> dict:
        session = state["session"]
        is_complete = bool(state.get("is_complete"))
        if not is_complete:
            return {}

        try:
            evaluation_report = await self._svc.evaluation_agent.evaluate_interview(
                getattr(session, "answers", []) or [],
                str(getattr(session, "session_id", state.get("session_id"))),
                api_key=state.get("api_key"),
            )
            try:
                self._svc._maybe_add_peer_learning_insight(evaluation_report)
            except Exception:
                pass

            session.evaluation_report = evaluation_report
            return {"evaluation_report": evaluation_report}
        except Exception as eval_error:
            logger.error(f"❌ Evaluation generation failed: {eval_error}", exc_info=True)
            return {"evaluation_error": str(eval_error), "evaluation_report": None}

    async def _n_build_response(self, state: _SubmitAnswerState) -> dict:
        session = state["session"]
        question_id = int(state.get("question_id") or 0)

        is_complete = bool(state.get("is_complete"))
        total_questions = len(getattr(session, "questions", []) or [])

        response: dict[str, Any] = {
            "transcript": state.get("transcript") or "",
            "metrics": state.get("metrics"),
            "micro_feedback": state.get("micro_feedback"),
            "complete": is_complete,
            "progress": self._svc.interviewer_agent.get_progress_indicator(question_id, total_questions),
            "requires_acknowledgment": (not is_complete),
            "current_question_id": question_id,
        }

        if is_complete:
            if "evaluation_report" in state:
                response["evaluation_report"] = state.get("evaluation_report")
            if "evaluation_error" in state:
                response["evaluation_error"] = state.get("evaluation_error")

        return {"response": response}
