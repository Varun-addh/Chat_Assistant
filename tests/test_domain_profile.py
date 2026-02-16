from app.utils.domain_profile import build_domain_profile, render_domain_hints


def test_domain_profile_video_streaming():
    p = build_domain_profile("Design a Prime Video like streaming platform with DRM and CDN")
    assert p.domain == "video_streaming"
    assert any("drm" in s.lower() for s in p.matched_signals) or "drm" in " ".join(p.nouns).lower()


def test_domain_profile_payments():
    p = build_domain_profile("Design a payment processing system with idempotency and settlement")
    assert p.domain == "payments_fintech"
    assert any("ledger" in n.lower() for n in p.nouns)


def test_domain_profile_chat():
    p = build_domain_profile("Design a real-time chat messaging app with WebSockets and delivery receipts")
    assert p.domain == "chat_messaging"
    assert any("message" in n.lower() for n in p.nouns)


def test_domain_profile_generic_fallback():
    p = build_domain_profile("Design a system")
    assert p.domain in {"generic", "api_platform"}  # allow minor evolution


def test_render_domain_hints_contains_domain_and_metrics():
    p = build_domain_profile("Design a ride-sharing dispatch system")
    hints = render_domain_hints(p, "REQUEST_FLOW")
    assert "DOMAIN HINTS" in hints
    assert "Metrics" in hints
