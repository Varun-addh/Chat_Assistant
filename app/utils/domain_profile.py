from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class DomainProfile:
    """Deterministic domain hints for system design prompting.

    This is intentionally lightweight (no LLM calls): we want stable behavior.
    """

    domain: str
    confidence: float
    matched_signals: List[str]

    # Hints are short lists; they should guide specificity without forcing buzzwords.
    nouns: List[str]
    critical_flows: List[str]
    key_metrics: List[str]
    failure_modes: List[str]
    suggested_components: List[str]

    # Cloud guidance is optional; never mix providers unless explicitly requested.
    cloud_provider_hint: str | None


_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    # Media
    "video_streaming": [
        "video", "stream", "streaming", "prime video", "netflix", "ott", "playback",
        "hls", "dash", "manifest", "drm", "cdn", "origin", "transcode", "encoding",
    ],
    "music_streaming": ["music", "spotify", "audio", "podcast"],

    # Social/Chat
    "chat_messaging": ["chat", "messaging", "dm", "direct message", "whatsapp", "signal", "telegram", "sms"],
    "social_feed": ["feed", "timeline", "followers", "likes", "posts", "comments", "social"],

    # Commerce/Payments
    "payments_fintech": ["payment", "payments", "stripe", "upi", "card", "wallet", "ledger", "settlement", "chargeback"],
    "marketplace_ecommerce": ["marketplace", "ecommerce", "cart", "checkout", "inventory", "orders", "shipping"],

    # Mobility
    "rideshare_delivery": ["ride", "rideshare", "uber", "ola", "driver", "rider", "dispatch", "eta", "delivery"],

    # Search/Ads
    "search_indexing": ["search", "index", "query", "ranking", "autocomplete"],
    "ads_bidding": ["ads", "ad", "bidding", "auction", "cpm", "cpc", "ctr", "impression"],

    # Infrastructure / Developer systems
    "storage_files": ["storage", "s3", "blob", "object store", "upload", "download"],
    "observability_platform": ["observability", "metrics", "logs", "tracing", "apm", "monitoring"],
    "api_platform": ["api gateway", "rate limit", "quota", "auth", "jwt", "oauth"],
}


_COMPANY_PROVIDER_HINTS: List[Tuple[str, str]] = [
    ("amazon", "AWS"),
    ("prime", "AWS"),
    ("aws", "AWS"),
    ("google", "GCP"),
    ("gcp", "GCP"),
    ("microsoft", "Azure"),
    ("azure", "Azure"),
]


def _score_domain(text: str) -> Tuple[str, float, List[str]]:
    t = (text or "").lower()
    if not t.strip():
        return "generic", 0.0, []

    best_domain = "generic"
    best_score = 0
    best_hits: List[str] = []

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        hits = [kw for kw in keywords if kw in t]
        score = len(hits)
        if score > best_score:
            best_domain = domain
            best_score = score
            best_hits = hits

    # Confidence is heuristic: 0..1, saturating quickly.
    conf = min(1.0, best_score / 6.0) if best_domain != "generic" else 0.1
    return best_domain, conf, best_hits


def _cloud_provider_hint(text: str) -> str | None:
    t = (text or "").lower()
    for needle, provider in _COMPANY_PROVIDER_HINTS:
        if needle in t:
            return provider
    return None


def build_domain_profile(system_description: str) -> DomainProfile:
    domain, confidence, signals = _score_domain(system_description)
    provider = _cloud_provider_hint(system_description)

    # Domain-specific hint packs. Keep them short and general.
    if domain == "video_streaming":
        return DomainProfile(
            domain=domain,
            confidence=confidence,
            matched_signals=signals,
            nouns=["playback", "catalog", "manifest", "DRM license", "CDN", "origin"],
            critical_flows=[
                "Browse catalog -> start playback -> fetch manifest -> fetch segments",
                "Entitlement check -> DRM license -> stream start",
                "Upload/ingest -> transcode -> package (HLS/DASH) -> publish",
            ],
            key_metrics=[
                "startup time (p95 ms)",
                "rebuffer ratio (%)",
                "CDN hit rate (%)",
                "origin egress (Gbps)",
            ],
            failure_modes=[
                "CDN outage / regional degradation",
                "license service latency spikes",
                "cache stampede on trending titles",
            ],
            suggested_components=[
                "CDN + origin shield",
                "playback session service",
                "catalog service + cache",
                "packager/transcoder pipeline",
                "DRM/license service",
            ],
            cloud_provider_hint=provider,
        )

    if domain == "payments_fintech":
        return DomainProfile(
            domain=domain,
            confidence=confidence,
            matched_signals=signals,
            nouns=["payment intent", "authorization", "capture", "ledger", "settlement"],
            critical_flows=[
                "Create payment intent -> authorize -> capture -> record ledger",
                "Refund/chargeback -> reversal -> reconciliation",
            ],
            key_metrics=["authorization success rate (%)", "p95 latency (ms)", "fraud rate (%)"],
            failure_modes=["gateway timeout", "duplicate submissions", "reconciliation drift"],
            suggested_components=["idempotency layer", "ledger store", "risk/fraud checks", "reconciliation jobs"],
            cloud_provider_hint=provider,
        )

    if domain == "chat_messaging":
        return DomainProfile(
            domain=domain,
            confidence=confidence,
            matched_signals=signals,
            nouns=["conversation", "message", "delivery receipt", "presence"],
            critical_flows=["Send message -> persist -> fanout -> deliver -> ack"],
            key_metrics=["p95 send latency (ms)", "delivery success rate (%)", "online users"],
            failure_modes=["fanout backlog", "offline delivery delays", "out-of-order delivery"],
            suggested_components=["websocket gateway", "message store", "fanout workers", "push notifications"],
            cloud_provider_hint=provider,
        )

    if domain == "rideshare_delivery":
        return DomainProfile(
            domain=domain,
            confidence=confidence,
            matched_signals=signals,
            nouns=["driver", "rider", "trip", "ETA", "dispatch"],
            critical_flows=["Request ride -> match -> accept -> start -> complete"],
            key_metrics=["match latency (ms)", "cancel rate (%)", "p95 ETA error (min)"],
            failure_modes=["dispatch hot-spot", "location staleness", "surge mismatch"],
            suggested_components=["matching service", "location pipeline", "pricing service", "trip store"],
            cloud_provider_hint=provider,
        )

    if domain == "marketplace_ecommerce":
        return DomainProfile(
            domain=domain,
            confidence=confidence,
            matched_signals=signals,
            nouns=["catalog", "inventory", "cart", "order", "shipment"],
            critical_flows=["Browse -> add to cart -> checkout -> reserve inventory -> place order"],
            key_metrics=["checkout success rate (%)", "p95 latency (ms)", "oversell incidents"],
            failure_modes=["inventory race", "payment failure", "cart expiration"],
            suggested_components=["inventory reservation", "order service", "payment integration", "fulfillment events"],
            cloud_provider_hint=provider,
        )

    if domain == "search_indexing":
        return DomainProfile(
            domain=domain,
            confidence=confidence,
            matched_signals=signals,
            nouns=["document", "index", "query", "ranking", "features"],
            critical_flows=["Ingest -> parse -> index -> query -> rank -> return"],
            key_metrics=["p95 query latency (ms)", "index freshness (s/min)", "qps"],
            failure_modes=["stale index", "hot shards", "relevance regressions"],
            suggested_components=["indexer workers", "query service", "feature store", "cache for popular queries"],
            cloud_provider_hint=provider,
        )

    # Generic fallback: still useful, but not buzzwordy.
    return DomainProfile(
        domain="generic",
        confidence=0.1,
        matched_signals=[],
        nouns=["users", "requests", "core entities", "state"],
        critical_flows=["Main user action -> validate -> persist -> respond"],
        key_metrics=["p95 latency (ms)", "throughput (rps)", "error rate (%)"],
        failure_modes=["downstream timeout", "cache stampede", "hot partition"],
        suggested_components=["API service", "database", "cache", "async worker (if needed)"],
        cloud_provider_hint=provider,
    )


def render_domain_hints(profile: DomainProfile, view_name: str) -> str:
    """Render a compact, prompt-safe hint block.

    Keep this short: it should steer the model, not overwhelm it.
    """

    vn = (view_name or "").upper().strip()
    provider_line = ""
    if profile.cloud_provider_hint:
        provider_line = (
            f"- Cloud: Prefer {profile.cloud_provider_hint} naming; do not mix providers unless explicitly asked.\n"
        )

    # View-specific emphasis (tiny tweaks, still generic).
    emphasis = ""
    if vn == "OBSERVABILITY":
        emphasis = "- Observability focus: instrumentation, signals, alerting, and runbooks (not business algorithms).\n"

    lines = [
        "DOMAIN HINTS (deterministic):",
        f"- Domain guess: {profile.domain} (confidence {profile.confidence:.2f})",
    ]
    if profile.matched_signals:
        lines.append(f"- Signals: {', '.join(profile.matched_signals[:8])}")
    if provider_line:
        lines.append(provider_line.rstrip("\n"))
    if emphasis:
        lines.append(emphasis.rstrip("\n"))

    lines.extend(
        [
            f"- Use domain nouns: {', '.join(profile.nouns)}",
            f"- Critical flows: {', '.join(profile.critical_flows[:2])}",
            f"- Metrics (pick 2-3): {', '.join(profile.key_metrics)}",
            f"- Failure modes (pick 1-2): {', '.join(profile.failure_modes)}",
            f"- Suggested components (use only if relevant): {', '.join(profile.suggested_components)}",
        ]
    )

    return "\n".join(lines).strip() + "\n"
