"""Regression test: auth middleware must run before rate limiting.

Bug: every logged-in user was metered as a demo user.

Starlette builds the middleware stack so the LAST registered runs FIRST (it is
the outermost wrapper). Auth was registered first and rate limiting second,
making rate limiting the outer layer. It reads request.state.user — which only
the auth middleware sets — so it saw None on every request:

    [DEMO_MODE] path=/api/question user_type=demo auth=False user_key_header=False
    Using environment GEMINI_API_KEY     <- the router's *registered* branch

...for a request carrying a valid JWT. Registered users were charged against
demo quota (30-minute window) rather than their tier, while the router served
them as registered. Confirmed against a live instance before and after the fix.

The ordering is invisible at the call site — swapping two adjacent lines in
main.py silently reintroduces it — so it is pinned here.
"""

from app.main import app


def _http_middleware_names():
    """Registered HTTP middleware, outermost first (execution order)."""
    names = []
    for mw in app.user_middleware:
        target = None
        kwargs = getattr(mw, "kwargs", {}) or {}
        if "dispatch" in kwargs:
            target = kwargs["dispatch"]
        else:
            for arg in (getattr(mw, "args", ()) or ()):
                if callable(arg):
                    target = arg
                    break
        if target is not None:
            names.append(getattr(target, "__name__", repr(target)))
        else:
            names.append(getattr(mw.cls, "__name__", repr(mw.cls)))
    return names


def test_auth_middleware_runs_before_rate_limit():
    """user_auth must be outermost relative to rate_limit.

    app.user_middleware is ordered outermost-first, so auth must appear at a
    LOWER index than rate limiting.
    """
    names = _http_middleware_names()
    assert "user_auth_middleware" in names, names
    assert "rate_limit_middleware" in names, names

    auth_at = names.index("user_auth_middleware")
    limit_at = names.index("rate_limit_middleware")

    assert auth_at < limit_at, (
        "rate limiting runs before authentication, so request.state.user is "
        "unset when it classifies the caller — every logged-in user gets "
        f"metered as demo. Execution order: {names}"
    )


def test_rate_limit_reads_user_from_request_state():
    """Pins the coupling that makes the ordering matter.

    If rate limiting stops depending on request.state.user, the ordering
    constraint above can be relaxed — but until then it must hold.
    """
    import inspect

    from app.middleware import rate_limit

    src = inspect.getsource(rate_limit.rate_limit_middleware)
    assert "request.state" in src and "user" in src, (
        "rate_limit_middleware no longer reads request.state.user; revisit "
        "whether the middleware ordering constraint is still required"
    )
