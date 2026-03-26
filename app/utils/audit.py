from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any
import asyncio
import logging

from app.utils.time import utcnow

logger = logging.getLogger(__name__)

# Fields that may contain user-generated content / PII.
# When audit_store_raw_text is False (the default), these are replaced with
# their SHA-256 hex digest so events remain analytically useful (dedup, join)
# without retaining the raw text.
_PII_FIELDS = frozenset({"question", "answer", "code", "message", "text", "content"})


def _scrub(record: Dict[str, Any], store_raw: bool) -> Dict[str, Any]:
	"""Return a copy of record with PII fields hashed if store_raw is False."""
	if store_raw:
		return record
	out: Dict[str, Any] = {}
	for k, v in record.items():
		if k in _PII_FIELDS and isinstance(v, str) and v:
			out[k] = "sha256:" + hashlib.sha256(v.encode("utf-8")).hexdigest()
		else:
			out[k] = v
	return out


class JsonlAuditor:
	def __init__(self, path: Optional[str] = None) -> None:
		self._path = Path(path) if path else None
		self._lock = asyncio.Lock()

	def configure(self, path: Optional[str]) -> None:
		self._path = Path(path) if path else None

	async def log(self, record: Dict[str, Any]) -> None:
		if not self._path:
			return

		# Import here to avoid circular imports at module load time.
		try:
			from app.config import settings as _settings
			store_raw: bool = bool(getattr(_settings, "analytics_store_raw_text", False))
		except Exception:
			store_raw = False

		safe_record = _scrub(record, store_raw)
		line = json.dumps({"ts": utcnow().isoformat(), **safe_record}, ensure_ascii=False)
		async with self._lock:
			self._path.parent.mkdir(parents=True, exist_ok=True)
			with self._path.open("a", encoding="utf-8") as f:
				f.write(line + "\n")

	async def prune(self, retain_days: int = 90) -> int:
		"""Remove audit log entries older than *retain_days*.

		Rewrites the log file by streaming through it line-by-line to avoid
		loading the entire file into memory.  Returns the number of lines pruned.
		If the log file doesn't exist or is empty, this is a no-op.
		"""
		if not self._path or not self._path.exists():
			return 0

		cutoff = datetime.now(timezone.utc) - timedelta(days=retain_days)
		pruned = 0
		tmp_path = self._path.with_suffix(".tmp")

		async with self._lock:
			try:
				with self._path.open("r", encoding="utf-8") as src, \
				     tmp_path.open("w", encoding="utf-8") as dst:
					for line in src:
						line = line.strip()
						if not line:
							continue
						try:
							obj = json.loads(line)
							ts_str = obj.get("ts", "")
							ts = datetime.fromisoformat(ts_str)
							if ts.tzinfo is None:
								ts = ts.replace(tzinfo=timezone.utc)
							if ts >= cutoff:
								dst.write(line + "\n")
							else:
								pruned += 1
						except Exception:
							# Malformed line — keep it to avoid silent data loss.
							dst.write(line + "\n")
			except Exception as e:
				logger.warning("audit prune: could not process %s: %s", self._path, e)
				try:
					tmp_path.unlink(missing_ok=True)
				except Exception:
					pass
				return 0

			try:
				import os
				os.replace(str(tmp_path), str(self._path))
			except Exception as e:
				logger.warning("audit prune: could not replace %s: %s", self._path, e)
				return 0

		if pruned:
			logger.info("audit prune: removed %d entries older than %d days from %s", pruned, retain_days, self._path)
		return pruned


auditor = JsonlAuditor()
