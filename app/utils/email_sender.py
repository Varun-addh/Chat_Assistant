"""Email sending utility (SMTP).

This module is intentionally small and dependency-free.
- In production: enable via EMAIL_ENABLED=true and configure SMTP_* settings.
- In dev/test: keep EMAIL_ENABLED=false and the app will log/return URLs instead.

We never log SMTP credentials.
"""

from __future__ import annotations

import asyncio
import smtplib
import ssl
from email.message import EmailMessage
import logging

from app.config import settings

logger = logging.getLogger(__name__)


def email_enabled() -> bool:
    return bool(getattr(settings, "email_enabled", False))


def send_email(*, to_email: str, subject: str, body_text: str) -> None:
    """Send a plaintext email.

    Raises RuntimeError if email is disabled or configuration is incomplete.
    """

    if not email_enabled():
        raise RuntimeError("Email is disabled (set EMAIL_ENABLED=true)")

    smtp_host = (getattr(settings, "smtp_host", None) or "").strip()
    smtp_port = int(getattr(settings, "smtp_port", 587) or 587)
    smtp_username = (getattr(settings, "smtp_username", None) or "").strip()
    smtp_password = (getattr(settings, "smtp_password", None) or "")
    smtp_use_tls = bool(getattr(settings, "smtp_use_tls", True))
    smtp_use_ssl = bool(getattr(settings, "smtp_use_ssl", False))
    from_email = (getattr(settings, "email_from", None) or "").strip()

    if not smtp_host:
        raise RuntimeError("SMTP_HOST is required when EMAIL_ENABLED=true")
    if not from_email:
        raise RuntimeError("EMAIL_FROM is required when EMAIL_ENABLED=true")
    if smtp_username and not smtp_password:
        raise RuntimeError("SMTP_PASSWORD is required when SMTP_USERNAME is set")

    msg = EmailMessage()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body_text)

    context = ssl.create_default_context()

    if smtp_use_ssl:
        with smtplib.SMTP_SSL(host=smtp_host, port=smtp_port, context=context, timeout=20) as server:
            if smtp_username:
                server.login(smtp_username, smtp_password)
            server.send_message(msg)
        return

    with smtplib.SMTP(host=smtp_host, port=smtp_port, timeout=20) as server:
        if smtp_use_tls:
            server.starttls(context=context)
        if smtp_username:
            server.login(smtp_username, smtp_password)
        server.send_message(msg)


def send_email_best_effort(*, to_email: str, subject: str, body_text: str) -> bool:
    """Try to send an email. Returns True on success, False otherwise."""

    try:
        send_email(to_email=to_email, subject=subject, body_text=body_text)
        return True
    except Exception as e:
        # Do not print body_text (may contain reset links).
        logger.warning("Email send failed (best effort): %s", e)
        return False


async def send_email_async(*, to_email: str, subject: str, body_text: str) -> None:
    """Async wrapper: runs blocking SMTP send in a thread to avoid freezing the event loop."""
    await asyncio.to_thread(send_email, to_email=to_email, subject=subject, body_text=body_text)


async def send_email_best_effort_async(*, to_email: str, subject: str, body_text: str) -> bool:
    """Async best-effort email wrapper."""
    try:
        await send_email_async(to_email=to_email, subject=subject, body_text=body_text)
        return True
    except Exception as e:
        logger.warning("Async email send failed (best effort): %s", e)
        return False
