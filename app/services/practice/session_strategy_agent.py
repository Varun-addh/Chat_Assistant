from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Optional

from app.config import settings
from app.database import get_db_context
from app.schemas import (
	PracticeInterviewQuestion,
	PracticeSession,
	QuestionDifficulty,
	SessionCoachingStyle,
	SessionFollowUpDepth,
	SessionStrategyAction,
	SessionStrategyDecision,
)
from app.services.chat.llm_service import llm_service
from app.services.practice.adaptive_pressure import adjust_difficulty, compute_pressure_state
from app.services.practice.learning_loops import compute_practice_insights
from app.services.practice.practice_progress import get_latest_next_session_plan
from app.services.practice.practice_scoring import score_answer

if TYPE_CHECKING:
	from app.schemas import AnswerSubmission
	from app.services.practice.practice_mode_service import PracticeModeService


logger = logging.getLogger(__name__)


def _harder_difficulty(base: Optional[QuestionDifficulty]) -> QuestionDifficulty:
	if base == QuestionDifficulty.EASY:
		return QuestionDifficulty.MEDIUM
	if base == QuestionDifficulty.MEDIUM:
		return QuestionDifficulty.HARD
	return QuestionDifficulty.HARD


def _easier_difficulty(base: Optional[QuestionDifficulty]) -> QuestionDifficulty:
	if base == QuestionDifficulty.HARD:
		return QuestionDifficulty.MEDIUM
	if base == QuestionDifficulty.MEDIUM:
		return QuestionDifficulty.EASY
	return QuestionDifficulty.EASY


def _max_follow_ups(total_questions: int) -> int:
	if total_questions <= 3:
		return 1
	if total_questions <= 7:
		return 2
	return 3


def _clean_reason(text: Any, fallback: str) -> str:
	value = str(text or fallback).strip()
	if not value:
		return fallback
	return value[:300]


class SessionStrategyTools:
	"""Small, explicit tool layer used by the bounded Session Brain."""

	def __init__(self, practice_service: PracticeModeService):
		self._svc = practice_service

	def get_learning_signal(self, *, session: PracticeSession) -> dict[str, Any]:
		user_id = getattr(session, "user_id", None)
		profile = getattr(session, "user_profile", None)
		domain = getattr(profile, "domain", None) if profile is not None else None
		if not user_id:
			return {
				"recommended_focus": [],
				"overall": {},
				"next_session_plan": None,
			}

		try:
			with get_db_context() as db:
				insights = compute_practice_insights(db, user_id=user_id, domain=domain)
				next_session_plan = get_latest_next_session_plan(db, user_id=user_id, domain=domain)
			return {
				"recommended_focus": list(insights.get("recommended_focus") or [])[:3],
				"overall": dict(insights.get("overall") or {}),
				"next_session_plan": next_session_plan,
			}
		except Exception as exc:
			logger.debug("Session brain learning signal unavailable: %s", exc)
			return {
				"recommended_focus": [],
				"overall": {},
				"next_session_plan": None,
			}

	async def ask_question(
		self,
		*,
		session: PracticeSession,
		next_index: int,
		target_difficulty: Optional[QuestionDifficulty],
		api_key: Optional[str],
		previous_question: Optional[PracticeInterviewQuestion] = None,
	) -> Optional[PracticeInterviewQuestion]:
		questions = list(getattr(session, "questions", []) or [])
		if next_index < 0 or next_index >= len(questions):
			return None

		existing = questions[next_index]
		resolved_difficulty = target_difficulty or getattr(existing, "difficulty", None) or getattr(session, "difficulty", None)
		if resolved_difficulty is None:
			resolved_difficulty = QuestionDifficulty.MEDIUM

		already_asked = [
			q.text.strip()
			for idx, q in enumerate(questions)
			if idx != next_index and getattr(q, "text", None)
		]

		# Add a topic-avoidance hint so the LLM picks a different sub-topic
		# (e.g. after DECREASE_DIFFICULTY on a topic the user bombed).
		if previous_question is not None:
			prev_cat = getattr(previous_question, "category", "")
			already_asked.append(previous_question.text.strip())
			# Also include key_points so the LLM avoids the same concept space
			for kp in (getattr(previous_question, "key_points", None) or []):
				already_asked.append(str(kp).strip())

		should_regenerate = bool(target_difficulty and getattr(existing, "difficulty", None) != target_difficulty)
		if not should_regenerate:
			return existing

		generated: Optional[PracticeInterviewQuestion] = None
		profile = getattr(session, "user_profile", None)
		if profile is not None:
			try:
				generated_list = await self._svc.adaptive_interviewer.generate_adaptive_questions(
					user_profile=profile,
					difficulty=resolved_difficulty,
					count=1,
					round_type=getattr(existing, "round_type", None) or getattr(session, "round_type", None),
					api_key=api_key,
					previously_asked=already_asked,
				)
				if generated_list:
					generated = generated_list[0]
			except Exception as exc:
				logger.debug("Session brain ask_question regeneration failed: %s", exc)

		if generated is None:
			try:
				candidates = self._svc.interviewer_agent.get_questions(
					difficulty=resolved_difficulty,
					count=max(3, len(questions)),
				)
				for candidate in candidates:
					if candidate.text.strip() not in already_asked:
						generated = candidate
						break
				if generated is None and candidates:
					generated = candidates[0]
			except Exception as exc:
				logger.debug("Session brain interviewer fallback failed: %s", exc)

		if generated is None:
			return existing

		generated.id = next_index + 1
		generated.round_type = getattr(generated, "round_type", None) or getattr(existing, "round_type", None) or getattr(session, "round_type", None)
		questions[next_index] = generated
		session.questions = questions
		session.difficulty = resolved_difficulty
		return generated

	async def generate_followup(
		self,
		*,
		session: PracticeSession,
		previous_question: PracticeInterviewQuestion,
		last_answer: AnswerSubmission,
		next_index: int,
		target_difficulty: Optional[QuestionDifficulty],
		follow_up_depth: SessionFollowUpDepth,
		coaching_style: SessionCoachingStyle,
		api_key: Optional[str],
	) -> Optional[PracticeInterviewQuestion]:
		questions = list(getattr(session, "questions", []) or [])
		already_asked = [q.text.strip() for q in questions if getattr(q, "text", None)]
		resolved_difficulty = target_difficulty or getattr(previous_question, "difficulty", None) or getattr(session, "difficulty", None)
		if resolved_difficulty is None:
			resolved_difficulty = QuestionDifficulty.MEDIUM

		follow_up = await self._svc.adaptive_interviewer.generate_follow_up_question(
			user_profile=getattr(session, "user_profile", None),
			difficulty=resolved_difficulty,
			round_type=getattr(previous_question, "round_type", None) or getattr(session, "round_type", None),
			previous_question=previous_question,
			transcript=getattr(last_answer, "transcript", "") or "",
			micro_feedback=getattr(last_answer, "micro_feedback", None),
			already_asked=already_asked,
			target_question_id=next_index + 1,
			follow_up_depth=follow_up_depth,
			coaching_style=coaching_style,
			api_key=api_key,
		)
		if follow_up is None:
			return None

		if 0 <= next_index < len(questions):
			questions[next_index] = follow_up
			session.questions = questions
		session.difficulty = resolved_difficulty
		return follow_up

	async def end_session(self, *, session: PracticeSession, api_key: Optional[str]) -> dict[str, Any]:
		return await self._svc.end_session(session.session_id, api_key=api_key)


class SessionStrategyAgent:
	"""A bounded planner that chooses exactly one next-step action per user turn."""

	def __init__(self, practice_service: PracticeModeService):
		self._svc = practice_service
		self.tools = SessionStrategyTools(practice_service)

	async def preview_next_action(
		self,
		*,
		session: PracticeSession,
		previous_question: PracticeInterviewQuestion,
		last_answer: AnswerSubmission,
		api_key: Optional[str],
	) -> SessionStrategyDecision:
		context = self._build_context(
			session=session,
			previous_question=previous_question,
			last_answer=last_answer,
		)
		decision = await self._decide(context=context, api_key=api_key)
		final_decision = self._apply_guardrails(context=context, decision=decision)
		self._log_decision(stage="preview", session_id=session.session_id, decision=final_decision)
		return final_decision

	async def execute_next_action(
		self,
		*,
		session: PracticeSession,
		previous_question: PracticeInterviewQuestion,
		last_answer: AnswerSubmission,
		decision: SessionStrategyDecision,
		api_key: Optional[str],
	) -> dict[str, Any]:
		self._log_decision(stage="execute", session_id=session.session_id, decision=decision)
		next_index = int(getattr(session, "current_question_index", 0)) + 1
		total_questions = len(getattr(session, "questions", []) or [])
		progress_current = min(len(getattr(session, "answers", []) or []), total_questions)

		if decision.action == SessionStrategyAction.END_SESSION or next_index >= total_questions:
			end_result = await self.tools.end_session(session=session, api_key=api_key)
			return {
				"complete": True,
				"evaluation_report": end_result.get("evaluation_report") or getattr(session, "evaluation_report", None),
				"progress": self._svc.interviewer_agent.get_progress_indicator(max(progress_current, 1), total_questions),
				"strategy": decision,
			}

		question: Optional[PracticeInterviewQuestion] = None
		if decision.action == SessionStrategyAction.FOLLOW_UP:
			question = await self.tools.generate_followup(
				session=session,
				previous_question=previous_question,
				last_answer=last_answer,
				next_index=next_index,
				target_difficulty=decision.target_difficulty,
				follow_up_depth=decision.follow_up_depth,
				coaching_style=decision.coaching_style,
				api_key=api_key,
			)

		if question is None:
			question = await self.tools.ask_question(
				session=session,
				next_index=next_index,
				target_difficulty=decision.target_difficulty,
				api_key=api_key,
				previous_question=previous_question,
			)

		if question is None:
			end_result = await self.tools.end_session(session=session, api_key=api_key)
			return {
				"complete": True,
				"evaluation_report": end_result.get("evaluation_report") or getattr(session, "evaluation_report", None),
				"progress": self._svc.interviewer_agent.get_progress_indicator(max(progress_current, 1), total_questions),
				"strategy": decision,
			}

		return {
			"complete": False,
			"next_question": question,
			"progress": self._svc.interviewer_agent.get_progress_indicator(int(getattr(question, "id", next_index + 1)), total_questions),
			"strategy": decision,
		}

	def _build_context(
		self,
		*,
		session: PracticeSession,
		previous_question: PracticeInterviewQuestion,
		last_answer: AnswerSubmission,
	) -> dict[str, Any]:
		mf = getattr(last_answer, "micro_feedback", None)
		metrics = getattr(last_answer, "metrics", None)
		answer_score = score_answer(answer=last_answer)
		learning_signal = self.tools.get_learning_signal(session=session)
		pressure = compute_pressure_state(session=session)
		current_difficulty = getattr(session, "difficulty", None) or getattr(previous_question, "difficulty", None) or QuestionDifficulty.MEDIUM
		total_questions = len(getattr(session, "questions", []) or [])
		remaining_questions = max(0, total_questions - len(getattr(session, "answers", []) or []))
		transcript = (getattr(last_answer, "transcript", "") or "").strip()
		transcript_word_count = len(transcript.split())
		recent_actions: list[str] = []
		executed_follow_ups = 0
		for item in list(getattr(session, "strategy_history", []) or []):
			action = getattr(item, "action", None)
			if action is None and isinstance(item, dict):
				action = item.get("action")
			if isinstance(action, SessionStrategyAction):
				action_value = action.value
			else:
				action_value = str(action or "").strip().upper()
			if not action_value:
				continue
			recent_actions.append(action_value)
			if action_value == SessionStrategyAction.FOLLOW_UP.value:
				executed_follow_ups += 1

		last_strategy = recent_actions[-1] if recent_actions else None
		follow_up_streak = 0
		for action_value in reversed(recent_actions):
			if action_value != SessionStrategyAction.FOLLOW_UP.value:
				break
			follow_up_streak += 1

		follow_up_budget = _max_follow_ups(total_questions)

		return {
			"session": {
				"session_id": session.session_id,
				"answered_questions": len(getattr(session, "answers", []) or []),
				"total_questions": total_questions,
				"remaining_questions": remaining_questions,
				"current_difficulty": current_difficulty.value if isinstance(current_difficulty, QuestionDifficulty) else str(current_difficulty),
				"current_coaching_style": getattr(session, "current_coaching_style", SessionCoachingStyle.BALANCED).value,
				"last_strategy_action": last_strategy,
				"recent_strategy_actions": recent_actions[-4:],
				"executed_follow_ups": executed_follow_ups,
				"follow_up_streak": follow_up_streak,
				"follow_up_budget": follow_up_budget,
			},
			"previous_question": {
				"id": previous_question.id,
				"text": previous_question.text,
				"category": previous_question.category,
				"difficulty": previous_question.difficulty.value,
			},
			"performance": {
				"overall_score": round(float(answer_score.overall_score), 1),
				"correctness_score": getattr(mf, "correctness_score", None),
				"technical_accuracy": getattr(mf, "technical_accuracy", None),
				"missed_key_points": list(getattr(mf, "key_points_missed", None) or [])[:3],
				"strengths": list(getattr(mf, "strengths", None) or [])[:2],
				"improvement_areas": list(getattr(mf, "improvement_areas", None) or [])[:2],
				"filler_count": getattr(metrics, "filler_count", None),
				"wpm": getattr(metrics, "wpm", None),
				"confidence_score": getattr(metrics, "confidence_score", None),
				"transcript_word_count": transcript_word_count,
				"transcript_excerpt": transcript[:280],
			},
			"learning_signal": learning_signal,
			"pressure": pressure,
		}

	async def _decide(self, *, context: dict[str, Any], api_key: Optional[str]) -> SessionStrategyDecision:
		if self._should_use_llm(api_key):
			decision = await self._llm_decision(context=context, api_key=api_key)
			if decision is not None:
				return decision
		return self._fallback_decision(context=context)

	def _should_use_llm(self, api_key: Optional[str]) -> bool:
		if not bool(getattr(settings, "enable_practice_session_brain_llm", True)):
			return False
		key = (api_key or "").strip()
		if len(key) < 20:
			return False
		return True

	async def _llm_decision(self, *, context: dict[str, Any], api_key: str) -> Optional[SessionStrategyDecision]:
		prompt = self._build_llm_prompt(context)
		try:
			text = await llm_service.generate_text(
				prompt=prompt,
				api_key=api_key,
				json_mode=True,
				temperature=0.1,
				max_tokens=350,
			)
			if not text:
				return None
			return self._parse_llm_decision(text=text, learning_focus=context.get("learning_signal", {}).get("recommended_focus") or [])
		except Exception as exc:
			logger.debug("Session brain LLM decision failed: %s", exc)
			return None

	def _build_llm_prompt(self, context: dict[str, Any]) -> str:
		serialized = json.dumps(context, ensure_ascii=True, default=str)
		return (
			"You are the Session Strategy Agent for a live mock interview.\n"
			"Choose exactly ONE next-step action for the next user turn.\n\n"
			"Decision priorities (highest to lowest):\n"
			"1. Maximize learning value from the very next turn.\n"
			"2. Avoid repetition and getting stuck in the same drill pattern.\n"
			"3. Avoid frustration by easing up when the answer was weak or pressure is high.\n"
			"4. Preserve interview realism by keeping the flow credible and time-bounded.\n\n"
			"Available actions/tools:\n"
			"- ASK_QUESTION: move to a fresh question at the current level\n"
			"- FOLLOW_UP: drill into the last answer before moving on\n"
			"- INCREASE_DIFFICULTY: move to a harder fresh question\n"
			"- DECREASE_DIFFICULTY: move to an easier fresh question\n"
			"- END_SESSION: end only if the session is effectively complete\n\n"
			"Hard rules:\n"
			"- Return one action only.\n"
			"- Keep the interview moving; avoid loops.\n"
			"- Respect session.follow_up_budget and recent_strategy_actions from context.\n"
			"- Prefer FOLLOW_UP only when the answer was mostly decent but shallow or missing a few key points.\n"
			"- Prefer DECREASE_DIFFICULTY over FOLLOW_UP when the answer was very weak or extremely short.\n"
			"- Use INCREASE_DIFFICULTY only when recent performance is strong.\n"
			"- Do not END_SESSION early unless there is no meaningful next question left.\n\n"
			f"Context JSON:\n{serialized}\n\n"
			"Return JSON only in this schema:\n"
			"{\n"
			'  "action": "ASK_QUESTION|FOLLOW_UP|INCREASE_DIFFICULTY|DECREASE_DIFFICULTY|END_SESSION",\n'
			'  "reason": "short reason",\n'
			'  "coaching_style": "supportive|balanced|challenging",\n'
			'  "follow_up_depth": "none|light|deep",\n'
			'  "target_difficulty": "easy|medium|hard|null"\n'
			"}"
		)

	def _parse_llm_decision(self, *, text: str, learning_focus: list[str]) -> Optional[SessionStrategyDecision]:
		payload = text.strip()
		if "```json" in payload:
			payload = payload.split("```json", 1)[1].split("```", 1)[0].strip()
		elif "```" in payload:
			payload = payload.split("```", 1)[1].split("```", 1)[0].strip()

		match = re.search(r"\{.*\}", payload, re.DOTALL)
		if match:
			payload = match.group(0).strip()

		try:
			data = json.loads(payload)
		except Exception:
			payload = re.sub(r",\s*([\]}])", r"\1", payload)
			try:
				data = json.loads(payload)
			except Exception:
				return None

		if not isinstance(data, dict):
			return None

		action_raw = str(data.get("action") or "ASK_QUESTION").strip().upper()
		style_raw = str(data.get("coaching_style") or "balanced").strip().lower()
		depth_raw = str(data.get("follow_up_depth") or "none").strip().lower()
		target_difficulty = data.get("target_difficulty")

		try:
			action = SessionStrategyAction(action_raw)
		except Exception:
			return None

		try:
			coaching_style = SessionCoachingStyle(style_raw)
		except Exception:
			coaching_style = SessionCoachingStyle.BALANCED

		try:
			follow_up_depth = SessionFollowUpDepth(depth_raw)
		except Exception:
			follow_up_depth = SessionFollowUpDepth.NONE

		resolved_difficulty = None
		if isinstance(target_difficulty, str) and target_difficulty.strip():
			try:
				resolved_difficulty = QuestionDifficulty(target_difficulty.strip().lower())
			except Exception:
				resolved_difficulty = None

		return SessionStrategyDecision(
			action=action,
			reason=_clean_reason(data.get("reason"), "Session brain chose the next step."),
			coaching_style=coaching_style,
			follow_up_depth=follow_up_depth,
			target_difficulty=resolved_difficulty,
			source="llm",
			learning_focus=[str(item).strip() for item in learning_focus if str(item).strip()][:3],
		)

	def _fallback_decision(self, *, context: dict[str, Any]) -> SessionStrategyDecision:
		session_ctx = context.get("session", {})
		perf = context.get("performance", {})
		pressure = context.get("pressure", {})
		current_difficulty = QuestionDifficulty(str(session_ctx.get("current_difficulty") or QuestionDifficulty.MEDIUM.value).lower())
		remaining_questions = int(session_ctx.get("remaining_questions") or 0)
		correctness = perf.get("correctness_score")
		transcript_word_count = int(perf.get("transcript_word_count") or 0)
		missed = list(perf.get("missed_key_points") or [])
		overall = float(perf.get("overall_score") or 0.0)
		mode = str(pressure.get("mode") or "balanced")
		learning_focus = [str(item).strip() for item in context.get("learning_signal", {}).get("recommended_focus") or [] if str(item).strip()][:3]

		if remaining_questions <= 0:
			return SessionStrategyDecision(
				action=SessionStrategyAction.END_SESSION,
				reason="The session has no remaining questions.",
				coaching_style=SessionCoachingStyle.BALANCED,
				follow_up_depth=SessionFollowUpDepth.NONE,
				target_difficulty=current_difficulty,
				source="fallback_rules",
				learning_focus=learning_focus,
			)

		if correctness is not None and (int(correctness) < 35 or transcript_word_count < 6):
			target = _easier_difficulty(current_difficulty)
			action = SessionStrategyAction.DECREASE_DIFFICULTY if target != current_difficulty else SessionStrategyAction.ASK_QUESTION
			return SessionStrategyDecision(
				action=action,
				reason="The last answer was too weak for a drill-down, so move to an easier fresh question.",
				coaching_style=SessionCoachingStyle.SUPPORTIVE,
				follow_up_depth=SessionFollowUpDepth.NONE,
				target_difficulty=target,
				source="fallback_rules",
				learning_focus=learning_focus,
			)

		if missed and overall >= 60.0:
			depth = SessionFollowUpDepth.DEEP if overall >= 80.0 else SessionFollowUpDepth.LIGHT
			style = SessionCoachingStyle.CHALLENGING if mode == "challenging" else SessionCoachingStyle.BALANCED
			return SessionStrategyDecision(
				action=SessionStrategyAction.FOLLOW_UP,
				reason="The answer is close enough to justify drilling into missed concepts before moving on.",
				coaching_style=style,
				follow_up_depth=depth,
				target_difficulty=current_difficulty,
				source="fallback_rules",
				learning_focus=learning_focus,
			)

		if overall >= 82.0:
			target = _harder_difficulty(current_difficulty)
			action = SessionStrategyAction.INCREASE_DIFFICULTY if target != current_difficulty else SessionStrategyAction.ASK_QUESTION
			return SessionStrategyDecision(
				action=action,
				reason="Recent performance is strong enough to raise the bar on the next fresh question.",
				coaching_style=SessionCoachingStyle.CHALLENGING,
				follow_up_depth=SessionFollowUpDepth.NONE,
				target_difficulty=target,
				source="fallback_rules",
				learning_focus=learning_focus,
			)

		style = SessionCoachingStyle.SUPPORTIVE if mode == "supportive" else SessionCoachingStyle.BALANCED
		return SessionStrategyDecision(
			action=SessionStrategyAction.ASK_QUESTION,
			reason="Keep the session moving with a fresh question at a stable difficulty.",
			coaching_style=style,
			follow_up_depth=SessionFollowUpDepth.NONE,
			target_difficulty=current_difficulty,
			source="fallback_rules",
			learning_focus=learning_focus,
		)

	def _apply_guardrails(self, *, context: dict[str, Any], decision: SessionStrategyDecision) -> SessionStrategyDecision:
		session_ctx = context.get("session", {})
		perf = context.get("performance", {})
		remaining_questions = int(session_ctx.get("remaining_questions") or 0)
		answered_questions = int(session_ctx.get("answered_questions") or 0)
		follow_up_budget = max(1, int(session_ctx.get("follow_up_budget") or 1))
		executed_follow_ups = max(0, int(session_ctx.get("executed_follow_ups") or 0))
		transcript_word_count = int(perf.get("transcript_word_count") or 0)
		correctness = perf.get("correctness_score")
		last_action = session_ctx.get("last_strategy_action")
		current_difficulty = QuestionDifficulty(str(session_ctx.get("current_difficulty") or QuestionDifficulty.MEDIUM.value).lower())

		if remaining_questions <= 0:
			return self._with_trace(
				context=context,
				proposed_decision=decision,
				guardrail="no_remaining_questions",
				decision=SessionStrategyDecision(
				action=SessionStrategyAction.END_SESSION,
				reason="Guardrail: no remaining questions.",
				coaching_style=decision.coaching_style,
				follow_up_depth=SessionFollowUpDepth.NONE,
				target_difficulty=decision.target_difficulty or current_difficulty,
				source="guardrail",
				learning_focus=decision.learning_focus,
				),
			)

		if decision.action == SessionStrategyAction.GIVE_FEEDBACK:
			return self._with_trace(
				context=context,
				proposed_decision=decision,
				guardrail="feedback_already_delivered",
				decision=SessionStrategyDecision(
				action=SessionStrategyAction.ASK_QUESTION,
				reason="Guardrail: feedback was already delivered this turn, so move to the next question.",
				coaching_style=decision.coaching_style,
				follow_up_depth=SessionFollowUpDepth.NONE,
				target_difficulty=decision.target_difficulty or current_difficulty,
				source="guardrail",
				learning_focus=decision.learning_focus,
				),
			)

		if decision.action == SessionStrategyAction.END_SESSION and answered_questions < max(2, min(3, int(session_ctx.get("total_questions") or 0))):
			return self._with_trace(
				context=context,
				proposed_decision=decision,
				guardrail="insufficient_signal_to_end",
				decision=SessionStrategyDecision(
				action=SessionStrategyAction.ASK_QUESTION,
				reason="Guardrail: do not end the session before collecting enough signal.",
				coaching_style=decision.coaching_style,
				follow_up_depth=SessionFollowUpDepth.NONE,
				target_difficulty=decision.target_difficulty or current_difficulty,
				source="guardrail",
				learning_focus=decision.learning_focus,
				),
			)

		if decision.action == SessionStrategyAction.FOLLOW_UP:
			if executed_follow_ups >= follow_up_budget:
				return self._with_trace(
					context=context,
					proposed_decision=decision,
					guardrail="follow_up_budget_exhausted",
					decision=SessionStrategyDecision(
						action=SessionStrategyAction.ASK_QUESTION,
						reason="Guardrail: the follow-up budget is already spent, so move to a fresh question.",
						coaching_style=decision.coaching_style,
						follow_up_depth=SessionFollowUpDepth.NONE,
						target_difficulty=decision.target_difficulty or current_difficulty,
						source="guardrail",
						learning_focus=decision.learning_focus,
					),
				)
			if last_action == SessionStrategyAction.FOLLOW_UP.value:
				return self._with_trace(
					context=context,
					proposed_decision=decision,
					guardrail="follow_up_chain_blocked",
					decision=SessionStrategyDecision(
					action=SessionStrategyAction.ASK_QUESTION,
					reason="Guardrail: avoid chaining follow-up after follow-up.",
					coaching_style=decision.coaching_style,
					follow_up_depth=SessionFollowUpDepth.NONE,
					target_difficulty=decision.target_difficulty or current_difficulty,
					source="guardrail",
					learning_focus=decision.learning_focus,
					),
				)
			if transcript_word_count < 8 or (correctness is not None and int(correctness) < 40):
				fallback_target = _easier_difficulty(current_difficulty)
				action = SessionStrategyAction.DECREASE_DIFFICULTY if fallback_target != current_difficulty else SessionStrategyAction.ASK_QUESTION
				return self._with_trace(
					context=context,
					proposed_decision=decision,
					guardrail="follow_up_thin_answer",
					decision=SessionStrategyDecision(
					action=action,
					reason="Guardrail: the last answer was too thin for a useful follow-up.",
					coaching_style=SessionCoachingStyle.SUPPORTIVE,
					follow_up_depth=SessionFollowUpDepth.NONE,
					target_difficulty=fallback_target,
					source="guardrail",
					learning_focus=decision.learning_focus,
					),
				)

		if decision.action == SessionStrategyAction.INCREASE_DIFFICULTY:
			decision = decision.model_copy(update={
				"target_difficulty": _harder_difficulty(current_difficulty),
			})

		if decision.action == SessionStrategyAction.DECREASE_DIFFICULTY:
			decision = decision.model_copy(update={
				"target_difficulty": _easier_difficulty(current_difficulty),
				"coaching_style": SessionCoachingStyle.SUPPORTIVE,
			})

		if decision.action in {SessionStrategyAction.ASK_QUESTION, SessionStrategyAction.FOLLOW_UP} and decision.target_difficulty is None:
			decision = decision.model_copy(update={"target_difficulty": current_difficulty})

		return self._with_trace(context=context, decision=decision)

	def _with_trace(
		self,
		*,
		context: dict[str, Any],
		decision: SessionStrategyDecision,
		proposed_decision: Optional[SessionStrategyDecision] = None,
		guardrail: Optional[str] = None,
	) -> SessionStrategyDecision:
		merged_trace = dict(getattr(decision, "decision_trace", None) or {})
		merged_trace.update(
			self._build_decision_trace(
				context=context,
				decision=decision,
				proposed_decision=proposed_decision or decision,
				guardrail=guardrail,
			)
		)
		return decision.model_copy(update={"decision_trace": merged_trace})

	def _build_decision_trace(
		self,
		*,
		context: dict[str, Any],
		decision: SessionStrategyDecision,
		proposed_decision: SessionStrategyDecision,
		guardrail: Optional[str],
	) -> dict[str, Any]:
		session_ctx = context.get("session", {})
		perf = context.get("performance", {})
		pressure = context.get("pressure", {})
		budget_max = max(1, int(session_ctx.get("follow_up_budget") or 1))
		budget_used = max(0, int(session_ctx.get("executed_follow_ups") or 0))
		return {
			"proposed_action": proposed_decision.action.value,
			"proposed_source": proposed_decision.source,
			"final_action": decision.action.value,
			"final_source": decision.source,
			"guardrail": guardrail,
			"overall_score": perf.get("overall_score"),
			"correctness_score": perf.get("correctness_score"),
			"transcript_word_count": perf.get("transcript_word_count"),
			"remaining_questions": session_ctx.get("remaining_questions"),
			"answered_questions": session_ctx.get("answered_questions"),
			"current_difficulty": session_ctx.get("current_difficulty"),
			"last_strategy_action": session_ctx.get("last_strategy_action"),
			"recent_strategy_actions": list(session_ctx.get("recent_strategy_actions") or []),
			"pressure_mode": pressure.get("mode"),
			"missed_key_points": list(perf.get("missed_key_points") or [])[:2],
			"follow_up_budget": {
				"used": budget_used,
				"max": budget_max,
				"remaining": max(0, budget_max - budget_used),
			},
		}

	def _log_decision(self, *, stage: str, session_id: str, decision: SessionStrategyDecision) -> None:
		trace = dict(getattr(decision, "decision_trace", None) or {})
		follow_up_budget = dict(trace.get("follow_up_budget") or {})
		logger.info(
			"Session brain %s | session=%s action=%s source=%s proposed=%s guardrail=%s score=%s correctness=%s remaining=%s followups=%s/%s",
			stage,
			session_id,
			decision.action.value,
			decision.source,
			trace.get("proposed_action"),
			trace.get("guardrail"),
			trace.get("overall_score"),
			trace.get("correctness_score"),
			trace.get("remaining_questions"),
			follow_up_budget.get("used"),
			follow_up_budget.get("max"),
		)