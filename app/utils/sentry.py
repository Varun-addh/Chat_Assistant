"""
Sentry integration for production error tracking and performance monitoring.
Automatically captures exceptions, performance data, and breadcrumbs.
"""
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
import logging

from app.config import settings


def init_sentry():
    """
    Initialize Sentry SDK for error tracking.
    Only enabled in production/staging environments with a valid DSN.
    """
    sentry_dsn = settings.sentry_dsn if hasattr(settings, 'sentry_dsn') else None
    
    if not sentry_dsn:
        logging.info("Sentry disabled: SENTRY_DSN not configured")
        return
    
    if settings.app_env.lower() in ("development", "test"):
        logging.info(f"Sentry disabled in {settings.app_env} environment")
        return
    
    # Determine sample rate based on environment
    traces_sample_rate = 0.1  # 10% of transactions in production
    if settings.app_env.lower() == "staging":
        traces_sample_rate = 0.5  # 50% in staging for better debugging
    
    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=settings.app_env,
        release=settings.app_version if hasattr(settings, 'app_version') else "1.0.0",
        
        # Integrations
        integrations=[
            FastApiIntegration(
                transaction_style="endpoint",  # Group by endpoint, not full URL
                failed_request_status_codes=[500, 501, 502, 503, 504],
            ),
            SqlalchemyIntegration(),
            AsyncioIntegration(),
            LoggingIntegration(
                level=logging.INFO,  # Capture info and above
                event_level=logging.ERROR  # Send errors to Sentry
            ),
        ],
        
        # Performance Monitoring
        traces_sample_rate=traces_sample_rate,
        
        # Profiles (optional - captures code-level performance)
        # profiles_sample_rate=0.1,  # Uncomment to enable profiling
        
        # Filter out sensitive data
        before_send=_before_send,
        
        # Additional options
        send_default_pii=False,  # Don't send personally identifiable information
        attach_stacktrace=True,
        max_breadcrumbs=50,
        
        # Custom tags for filtering in Sentry UI
        _experiments={
            "profiles_sample_rate": 0.1,  # Enable profiling at 10%
        }
    )
    
    # Set custom context
    sentry_sdk.set_tag("app_name", settings.app_name)
    sentry_sdk.set_tag("llm_provider", settings.llm_provider)
    
    logging.info(f"✅ Sentry initialized for {settings.app_env} environment")


def _before_send(event, hint):
    """
    Filter/modify events before sending to Sentry.
    Use this to scrub sensitive data or ignore known issues.
    """
    # Don't send events for specific exceptions you want to ignore
    if "exc_info" in hint:
        exc_type, exc_value, tb = hint["exc_info"]
        
        # Ignore specific exception types
        if isinstance(exc_value, (KeyboardInterrupt, SystemExit)):
            return None
        
        # Add custom fingerprinting for better grouping
        # Example: Group all "API key missing" errors together
        if "api key" in str(exc_value).lower():
            event["fingerprint"] = ["api-key-error"]
    
    # Scrub sensitive data from request headers
    if "request" in event:
        if "headers" in event["request"]:
            headers = event["request"]["headers"]
            # Remove sensitive headers (case-insensitive lookup)
            headers_lower = {k.lower(): k for k in headers.keys()}
            for sensitive_header in ["authorization", "cookie", "x-api-key"]:
                orig_key = headers_lower.get(sensitive_header)
                if orig_key is not None:
                    headers[orig_key] = "[REDACTED]"
    
    # Scrub API keys from error messages
    if "message" in event:
        message = event["message"]
        # Simple pattern matching - improve as needed
        if "gsk_" in message or "AIza" in message:
            event["message"] = "[API key redacted from error message]"
    
    return event


def capture_exception(error: Exception, context: dict = None):
    """
    Manually capture an exception with additional context.
    
    Usage:
        try:
            risky_operation()
        except Exception as e:
            capture_exception(e, {"user_id": user.id, "operation": "risky_operation"})
    """
    if context:
        sentry_sdk.set_context("custom", context)
    
    sentry_sdk.capture_exception(error)


def set_user_context(user_id: str = None, email: str = None, username: str = None):
    """
    Set user context for error tracking.
    Helps identify which users are experiencing issues.
    """
    sentry_sdk.set_user({
        "id": user_id,
        "email": email,
        "username": username
    })
