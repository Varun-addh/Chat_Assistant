from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResponsePlan:
	"""A small, explicit response routing contract.

	This object is meant to be created deterministically from request context
	(UI/API inputs + simple heuristics) and then threaded through prompt building.
	
	- intent: high-level category (e.g., 'system_design', 'coding', 'general')
	- depth: 'quick' | 'standard' | 'deep'
	- format: output shape hint (e.g., 'text', 'code', 'json')
	"""

	intent: str
	depth: str
	format: str
