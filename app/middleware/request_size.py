from __future__ import annotations

import re

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import settings


class RequestBodyTooLarge(RuntimeError):
    pass


class RequestSizeLimitMiddleware:
    """Enforce a global HTTP body limit with a higher cap for practice media uploads."""

    _practice_media_pattern = re.compile(r"^/api/practice/session/[^/]+/media$")

    def __init__(
        self,
        app: ASGIApp,
        *,
        default_limit_bytes: int | None = None,
        practice_media_limit_bytes: int | None = None,
    ) -> None:
        self.app = app
        self.default_limit_bytes = int(default_limit_bytes or settings.max_request_body_bytes or 0)
        self.practice_media_limit_bytes = int(
            practice_media_limit_bytes or settings.max_practice_media_upload_bytes or self.default_limit_bytes
        )

    def _limit_for_path(self, path: str) -> int:
        if self._practice_media_pattern.fullmatch(path):
            return self.practice_media_limit_bytes
        return self.default_limit_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        limit = self._limit_for_path(path)
        if limit <= 0:
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        content_length = headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > limit:
                    response = JSONResponse(
                        {"detail": f"Request body too large (max {limit // (1024 * 1024)}MB)"},
                        status_code=413,
                    )
                    await response(scope, receive, send)
                    return
            except ValueError:
                pass

        consumed = 0

        async def limited_receive() -> Message:
            nonlocal consumed
            message = await receive()
            if message.get("type") != "http.request":
                return message

            body = message.get("body", b"") or b""
            consumed += len(body)
            if consumed > limit:
                raise RequestBodyTooLarge(f"Request body exceeds {limit} bytes")
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyTooLarge:
            response = JSONResponse(
                {"detail": f"Request body too large (max {limit // (1024 * 1024)}MB)"},
                status_code=413,
            )
            await response(scope, receive, send)