from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple


def _norm_items(items: Any, *, limit: int = 12) -> list[str]:
	if items is None:
		return []
	if isinstance(items, (str, int, float, bool)):
		items = [items]
	if not isinstance(items, list):
		items = [items]

	out: list[str] = []
	for it in items:
		s = str(it).strip()
		if not s:
			continue
		out.append(" ".join(s.split()))
		if len(out) >= limit:
			break
	return out


def _norm_question(q: str) -> str:
	return " ".join((q or "").strip().lower().split())


@dataclass(frozen=True)
class MirrorProgress:
	confidence_prev: Optional[float]
	confidence_curr: Optional[float]
	confidence_delta: Optional[float]
	gaps_closed: tuple[str, ...]
	new_gaps: tuple[str, ...]
	new_strengths: tuple[str, ...]
	new_red_flags: tuple[str, ...]
	red_flags_resolved: tuple[str, ...]


def compute_mirror_progress(prev_report: dict[str, Any], curr_report: dict[str, Any]) -> MirrorProgress:
	prev_gaps = set(_norm_items(prev_report.get("gaps"), limit=20))
	curr_gaps = set(_norm_items(curr_report.get("gaps"), limit=20))

	prev_strengths = set(_norm_items(prev_report.get("strengths"), limit=20))
	curr_strengths = set(_norm_items(curr_report.get("strengths"), limit=20))

	prev_red = set(_norm_items(prev_report.get("red_flags"), limit=20))
	curr_red = set(_norm_items(curr_report.get("red_flags"), limit=20))

	def _f(v: Any) -> Optional[float]:
		try:
			if v is None:
				return None
			return float(v)
		except Exception:
			return None

	c_prev = _f(prev_report.get("confidence"))
	c_curr = _f(curr_report.get("confidence"))
	c_delta = (c_curr - c_prev) if (c_prev is not None and c_curr is not None) else None

	gaps_closed = tuple(sorted(prev_gaps - curr_gaps))
	new_gaps = tuple(sorted(curr_gaps - prev_gaps))
	new_strengths = tuple(sorted(curr_strengths - prev_strengths))
	new_red_flags = tuple(sorted(curr_red - prev_red))
	red_flags_resolved = tuple(sorted(prev_red - curr_red))

	return MirrorProgress(
		confidence_prev=c_prev,
		confidence_curr=c_curr,
		confidence_delta=c_delta,
		gaps_closed=gaps_closed,
		new_gaps=new_gaps,
		new_strengths=new_strengths,
		new_red_flags=new_red_flags,
		red_flags_resolved=red_flags_resolved,
	)


def format_mirror_progress_markdown(progress: MirrorProgress, *, max_items: int = 3) -> str:
	"""Return a compact markdown block describing progress.

	Returns "" if there isn't anything meaningful to report.
	"""
	lines: list[str] = []

	if progress.confidence_prev is not None and progress.confidence_curr is not None:
		delta = progress.confidence_delta
		if delta is None:
			conf_line = f"- Confidence: {progress.confidence_prev:.2f} → {progress.confidence_curr:.2f}"
		else:
			sign = "+" if delta >= 0 else ""
			conf_line = (
				f"- Confidence: {progress.confidence_prev:.2f} → {progress.confidence_curr:.2f} ({sign}{delta:.2f})"
			)
		lines.append(conf_line)

	def _fmt(label: str, items: Iterable[str]) -> None:
		items_list = list(items)
		if not items_list:
			return
		shown = items_list[:max_items]
		more = len(items_list) - len(shown)
		suffix = f" (+{more} more)" if more > 0 else ""
		lines.append(f"- {label}: " + "; ".join(shown) + suffix)

	_fmt("Gaps closed", progress.gaps_closed)
	_fmt("New gaps", progress.new_gaps)
	_fmt("New strengths", progress.new_strengths)
	_fmt("New red flags", progress.new_red_flags)
	_fmt("Red flags resolved", progress.red_flags_resolved)

	if not lines:
		return ""

	return "### Progress since your last draft\n" + "\n".join(lines)


def find_previous_mirror_attempt(
	*,
	question: str,
	mirror_history: list[dict[str, Any]],
	skip_latest: bool = False,
) -> Optional[dict[str, Any]]:
	"""Return the most recent mirror_history item matching the question."""
	qk = _norm_question(question)
	items = mirror_history[:-1] if (skip_latest and mirror_history) else mirror_history
	for it in reversed(items):
		try:
			if _norm_question(str(it.get("question") or "")) == qk:
				return it
		except Exception:
			continue
	return None
