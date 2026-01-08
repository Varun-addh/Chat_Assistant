from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class LayerDef:
    """Defines one layer in a 5-layer architecture view."""

    title: str
    responsibility: str


# Single source of truth for the 5-layer model used across multi-view outputs.
#
# IMPORTANT:
# - Keys are UPPERCASE view names used by `enforce_story_contract()` and by
#   `ArchitectureViewType.<NAME>.name`.
# - Layer TITLES are stable to keep the UI predictable and interview-ready.
# - The LLM is responsible for making the *content* domain-specific.
LAYER_MODEL_BY_VIEW: Dict[str, List[LayerDef]] = {
    "REQUEST_FLOW": [
        LayerDef("Client & Auth", "UI, login, token issuance, session context"),
        LayerDef("API Gateway", "Routing, auth validation, rate limiting"),
        LayerDef(
            "Core Business Logic",
            "Main workflow: validation, rules, conflict checks, decisions",
        ),
        LayerDef("Persistence Layer", "DB reads/writes, concurrency control, consistency guarantees"),
        LayerDef("Async & Sync Layer", "Notifications, external syncs, background jobs, retries"),
    ],
    "DATA_MODEL": [
        LayerDef("Primary Data Store", "Source of truth for core entities and transactions"),
        LayerDef("Cache Layer", "Low-latency reads and hot data acceleration"),
        LayerDef("Index & Search Layer", "Search/indexing for queries not served by the primary DB"),
        LayerDef("Analytics Layer", "Aggregations, reporting, OLAP/warehouse (batch/stream)"),
        LayerDef("Backup & Recovery Layer", "Replication, snapshots, restore, DR (RPO/RTO)"),
    ],
    "DEPLOYMENT": [
        LayerDef("Global Edge & Routing", "CDN, Anycast/DNS, DDoS, geo-routing"),
        LayerDef("API Edge & Security", "Gateway/WAF, auth, rate limiting, request shaping"),
        LayerDef("Compute & Orchestration", "Kubernetes/VMs, autoscaling, service discovery/mesh"),
        LayerDef("Async Processing", "Queues/brokers, workers, schedulers, DLQ"),
        LayerDef("Persistence & DR", "Databases, backups, replication, disaster recovery"),
    ],
    "OBSERVABILITY": [
        LayerDef("Metrics & KPIs", "SLIs/SLOs, dashboards, latency/error saturation monitoring"),
        LayerDef("Centralized Logging", "Structured logs, correlation IDs, log search/retention"),
        LayerDef("Distributed Tracing", "Spans, trace propagation, bottleneck/root cause analysis"),
        LayerDef("Alerting & Incident Response", "Alert rules, paging, runbooks, incident handling"),
        LayerDef("Reliability Governance", "On-call, postmortems, error budgets, reliability improvements"),
    ],
    "ASYNC_PROCESSING": [
        LayerDef("Producers", "Idempotent event emission from the core workflow"),
        LayerDef("Event Bus", "Queue/topic/broker providing buffering and delivery semantics"),
        LayerDef("Workers", "Consumers doing background computation and fanout"),
        LayerDef("Side Effects", "External integrations, notifications, search indexing, email/SMS"),
        LayerDef("Recovery & Reconciliation", "Retries, DLQ, replay, backfills, consistency repair"),
    ],
}


def get_layer_model(view_name: str) -> List[LayerDef]:
    """Return the canonical 5-layer model for a view.

    Args:
        view_name: e.g. 'REQUEST_FLOW', 'DATA_MODEL', 'DEPLOYMENT', 'OBSERVABILITY'.

    Returns:
        List of LayerDef. Empty list when the view is not layered.
    """

    key = (view_name or "").upper().strip()
    return list(LAYER_MODEL_BY_VIEW.get(key, []))


def get_layer_titles(view_name: str) -> List[str]:
    return [layer.title for layer in get_layer_model(view_name)]
