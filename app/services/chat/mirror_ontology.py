from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Optional

import hashlib
import json
import time
import asyncio

from app.config import settings
from app.services.core.redis_client import get_redis, redis_enabled


@dataclass(frozen=True)
class MirrorOntology:
	"""LLM-inferred expectations for a given interview question.

	This intentionally contains no hardcoded topic maps.
	"""
	topic: str
	primitives: tuple[str, ...]
	senior_signals: tuple[str, ...]
	red_flags: tuple[str, ...]
	likely_followups: tuple[str, ...]


GenerateTextFn = Callable[..., Awaitable[str]]


class MirrorOntologyGenerator:
	"""LLM-driven ontology generator with simple TTL/LRU caching."""

	def __init__(self, *, ttl_seconds: int = 60 * 60, max_entries: int = 256) -> None:
		self._ttl_seconds = max(60, int(ttl_seconds))
		self._max_entries = max(16, int(max_entries))
		# key -> (created_ts, MirrorOntology)
		self._cache: "OrderedDict[str, tuple[float, MirrorOntology]]" = OrderedDict()
		self._cache_lock = asyncio.Lock()

	def _cache_key(self, question: str) -> str:
		return " ".join((question or "").strip().lower().split())

	def _redis_key(self, key: str) -> str:
		# Keep keys short and safe for Redis.
		h = hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()
		prefix = (getattr(settings, "redis_key_prefix", "stratax") or "stratax").strip() or "stratax"
		return f"{prefix}:mirror:ontology:{h}"

	async def _get_cached_redis(self, key: str) -> Optional[MirrorOntology]:
		if not redis_enabled():
			return None
		try:
			r = await get_redis()
			if r is None:
				return None
			val = await r.get(self._redis_key(key))
			if not val:
				return None
			obj = json.loads(val)
			ontology = MirrorOntology(
				topic=str(obj.get("topic") or "General").strip() or "General",
				primitives=self._coerce_list(obj.get("primitives"), limit=8),
				senior_signals=self._coerce_list(obj.get("senior_signals"), limit=8),
				red_flags=self._coerce_list(obj.get("red_flags"), limit=6),
				likely_followups=self._coerce_list(obj.get("likely_followups"), limit=6),
			)
			return ontology
		except Exception:
			# Fail-open. Mirror should still work without Redis.
			return None

	async def _set_cached_redis(self, key: str, ontology: MirrorOntology) -> None:
		if not redis_enabled():
			return
		try:
			r = await get_redis()
			if r is None:
				return
			ttl = int(getattr(settings, "redis_cache_ttl_seconds", self._ttl_seconds) or self._ttl_seconds)
			ttl = max(60, ttl)
			payload = json.dumps(
				{
					"topic": ontology.topic,
					"primitives": list(ontology.primitives),
					"senior_signals": list(ontology.senior_signals),
					"red_flags": list(ontology.red_flags),
					"likely_followups": list(ontology.likely_followups),
				},
				ensure_ascii=False,
			)
			# setex = atomic set + TTL.
			await r.setex(self._redis_key(key), ttl, payload)
		except Exception:
			return

	def _get_cached(self, key: str) -> Optional[MirrorOntology]:
		item = self._cache.get(key)
		if not item:
			return None
		created_ts, ontology = item
		if (time.time() - created_ts) > self._ttl_seconds:
			# Expired
			try:
				del self._cache[key]
			except Exception:
				pass
			return None
		# Refresh LRU
		self._cache.move_to_end(key)
		return ontology

	def _set_cached(self, key: str, ontology: MirrorOntology) -> None:
		self._cache[key] = (time.time(), ontology)
		self._cache.move_to_end(key)
		while len(self._cache) > self._max_entries:
			self._cache.popitem(last=False)

	def _coerce_list(self, v: Any, *, limit: int) -> tuple[str, ...]:
		items: list[Any]
		if v is None:
			items = []
		elif isinstance(v, list):
			items = v
		else:
			items = [v]
		out: list[str] = []
		for it in items:
			s = str(it).strip()
			if not s:
				continue
			out.append(s)
			if len(out) >= limit:
				break
		return tuple(out)

	async def get(
		self,
		*,
		question: str,
		generate_text: GenerateTextFn,
		api_key: Optional[str] = None,
	) -> MirrorOntology:
		q = (question or "").strip()
		key = self._cache_key(q)

		async with self._cache_lock:
			cached = self._get_cached(key)
			if cached is not None:
				return cached

		# L2: Redis (cross-worker + survives restart)
		redis_cached = await self._get_cached_redis(key)
		if redis_cached is not None:
			async with self._cache_lock:
				self._set_cached(key, redis_cached)
			return redis_cached

		system_prompt = (
			"You generate an ontology for interview evaluation. "
			"Return ONLY a JSON object with keys: topic, primitives, senior_signals, red_flags, likely_followups. "
			"No extra keys. No prose."
		)
		user_prompt = (
			"Interview question:\n"
			+ q
			+ "\n\n"
			"Infer the most likely topic and the key concepts a strong candidate answer should cover. "
			"Be general and interview-relevant (avoid vendor-specific stacks unless the question implies them). "
			"If something is implied, don't over-penalize it."
		)

		raw = await generate_text(
			user_prompt,
			system_prompt=system_prompt,
			api_key=api_key,
			json_mode=True,
			temperature=0.2,
			max_tokens=500,
		)
		try:
			obj = json.loads((raw or "").strip())
		except Exception:
			obj = {}

		ontology = MirrorOntology(
			topic=str(obj.get("topic") or "General").strip() or "General",
			primitives=self._coerce_list(obj.get("primitives"), limit=8),
			senior_signals=self._coerce_list(obj.get("senior_signals"), limit=8),
			red_flags=self._coerce_list(obj.get("red_flags"), limit=6),
			likely_followups=self._coerce_list(obj.get("likely_followups"), limit=6),
		)

		async with self._cache_lock:
			self._set_cached(key, ontology)
		await self._set_cached_redis(key, ontology)
		return ontology
