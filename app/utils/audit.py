from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import asyncio


class JsonlAuditor:
	def __init__(self, path: Optional[str] = None) -> None:
		self._path = Path(path) if path else None
		self._lock = asyncio.Lock()

	def configure(self, path: Optional[str]) -> None:
		self._path = Path(path) if path else None

	async def log(self, record: Dict[str, Any]) -> None:
		if not self._path:
			return
		line = json.dumps({"ts": datetime.utcnow().isoformat(), **record}, ensure_ascii=False)
		async with self._lock:
			self._path.parent.mkdir(parents=True, exist_ok=True)
			with self._path.open("a", encoding="utf-8") as f:
				f.write(line + "\n")


auditor = JsonlAuditor()
