
from __future__ import annotations

import logging
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)


class ArchitectureViewType(str, Enum):
    """Types of architecture views - each tells one story."""
    
    SYSTEM_OVERVIEW = "system_overview"
    """High-level building blocks - what the system is made of"""
    
    REQUEST_FLOW = "request_flow"
    """Critical business path - what happens during user action"""
    
    ASYNC_PROCESSING = "async_processing"
    """Background & event handling - scalability & non-blocking behavior"""
    
    DATA_MODEL = "data_model"
    """Storage & persistence - where data lives and why"""
    
    DEPLOYMENT = "deployment"
    """Infrastructure & scaling - how it's deployed and scaled"""
    
    OBSERVABILITY = "observability"
    """Monitoring & reliability - how we keep it healthy"""
    
    SECURITY = "security"
    """Authentication, authorization, and protection"""


class DiagramStyle(str, Enum):
    """Visual style presets for diagrams."""
    
    MODERN = "modern"
    """Clean, professional, enterprise-ready"""
    
    MINIMAL = "minimal"
    """Simple, focused, no decorations"""
    
    DETAILED = "detailed"
    """Rich annotations and explanations"""


class ArchitectureView(BaseModel):
    """A single architectural view with its diagram and metadata."""
    
    view_type: ArchitectureViewType
    title: str = Field(..., description="Human-readable title")
    description: str = Field(..., description="What this view explains")
    mermaid_code: str = Field(..., description="Mermaid diagram code")
    key_insights: List[str] = Field(default_factory=list, description="3-5 key takeaways")
    complexity_level: str = Field(..., description="junior|mid|senior|architect")
    estimated_explanation_time: str = Field(..., description="How long to explain (e.g., '2 min')")
    

class ArchitecturePackage(BaseModel):
    """Complete architecture with multiple coordinated views."""
    
    system_name: str = Field(..., description="Name of the system being designed")
    description: str = Field(..., description="Brief system description")
    views: List[ArchitectureView] = Field(..., description="All architectural views")
    view_order: List[ArchitectureViewType] = Field(..., description="Recommended viewing order")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    total_views: int = Field(..., description="Number of views generated")
    

class ViewGenerationPrompt(BaseModel):
    """Prompt configuration for generating a specific view."""
    
    view_type: ArchitectureViewType
    system_prompt: str
    user_prompt_template: str
    constraints: List[str] = Field(default_factory=list)
    max_nodes: int = Field(default=15, description="Maximum nodes in diagram")
    max_edges: int = Field(default=20, description="Maximum connections")
    

class ArchitectureGeneratorService:
    """
    Service for generating multi-view architecture diagrams.
    
    This is the core intelligence that transforms a system description
    into multiple coordinated, professional architecture views.
    """
    
    def __init__(self):
        self.view_prompts = self._initialize_view_prompts()
        logger.info(f"✅ [ArchitectureGenerator] Service initialized with {len(self.view_prompts)} view types")

    def _estimate_problem_complexity(self, system_description: str) -> tuple[int, str]:
        """Heuristic complexity estimator for adaptive diagram sizing.

        We intentionally keep this lightweight and deterministic (no LLM call).
        Returns: (score, tier) where tier in {simple, medium, complex}.
        """
        text = (system_description or "").lower()
        if not text.strip():
            return 0, "simple"

        # Signal keywords: each indicates additional constraints/components.
        # Configurable via Settings for easy tuning without code changes.
        signals = list(getattr(settings, "architecture_complexity_signals", []) or [])

        score = 0
        for kw in signals:
            if kw in text:
                score += 1

        # Longer prompts often include more requirements.
        if len(text) > 120:
            score += 1
        if len(text) > 240:
            score += 1

        if score >= 9:
            return score, "complex"
        if score >= 4:
            return score, "medium"
        return score, "simple"

    def _dynamic_limits(self, view_type: ArchitectureViewType, system_description: str) -> tuple[int, int]:
        """Compute adaptive (max_nodes, max_edges) based on requirements complexity.

        The goal is *not* to explode diagram size, but to allow slightly more detail
        when the question clearly demands it.
        """
        prompt_config = self.view_prompts.get(view_type)
        if not prompt_config:
            return 15, 20

        base_nodes = int(prompt_config.max_nodes)
        base_edges = int(prompt_config.max_edges)

        score, tier = self._estimate_problem_complexity(system_description)
        if tier == "simple":
            mult = 0.9
        elif tier == "medium":
            mult = 1.1
        else:
            mult = 1.25

        nodes = max(6, int(round(base_nodes * mult)))
        edges = max(8, int(round(base_edges * mult)))

        # Keep System Overview tight no matter what (readability > completeness).
        if view_type == ArchitectureViewType.SYSTEM_OVERVIEW:
            nodes = min(nodes, 10)
            edges = min(edges, 12)

        # Hard caps to avoid renderer failures / unreadable diagrams.
        nodes = min(nodes, 22)
        edges = min(edges, 30)

        logger.info(
            "📐 Dynamic limits for %s: tier=%s score=%s => max_nodes=%s max_edges=%s",
            getattr(view_type, "value", str(view_type)),
            tier,
            score,
            nodes,
            edges,
        )

        return nodes, edges

    def _initialize_view_prompts(self) -> Dict[ArchitectureViewType, ViewGenerationPrompt]:
        """Initialize expert-crafted prompts for each view type."""
        
        return {
            ArchitectureViewType.SYSTEM_OVERVIEW: ViewGenerationPrompt(
                view_type=ArchitectureViewType.SYSTEM_OVERVIEW,
                system_prompt="""VIEW: SYSTEM OVERVIEW

OUTPUT CONTRACT (MANDATORY):
1) Output EXACTLY one Mermaid code block first, fenced as ```mermaid ... ```.
2) Immediately after, output ONLY this content (NO markdown headings like #/##):

- (exactly 5 bullets, each 1 sentence; each bullet on its own line and starts with '- ')
Goal: (one sentence)

NO other headings/sections. No tables. No code blocks other than Mermaid.

Mermaid rules (CRITICAL - VIOLATING THESE BREAKS RENDERING):
- flowchart only (flowchart TD or LR) - NO OTHER DIAGRAM TYPES
- ASCII only - NO unicode characters
- ARROWS: Use --> for connections (NOT -> which is invalid)
- Keep to <= 8 nodes and <= 10 edges
- NO init blocks (%%{init...}%%)
- NO classDef, NO linkStyle, NO class assignments (:::)
- NO CSS (@import, @keyframes, #container, #mermaid-svg)
- NO style tags or HTML
- Prefer no subgraphs
- Short node labels (2-3 words)
- ONLY output pure Mermaid flowchart syntax
""",
                user_prompt_template="""System: {system_description}

**CRITICAL**: All explanations must be SPECIFIC to "{system_description}". 
Use the actual features, use cases, and domain language of THIS SPECIFIC SYSTEM.
Do NOT use generic terms like "data" or "requests" - use the system's actual terminology.

Generate a simple, interview-friendly overview for THIS system.
""",
                constraints=[
                    "Maximum 8 nodes",
                    "Maximum 10 edges",
                    "No technology-specific names",
                    "No subgraphs unless absolutely necessary"
                ],
                max_nodes=8,
                max_edges=10
            ),
            
            ArchitectureViewType.REQUEST_FLOW: ViewGenerationPrompt(
                view_type=ArchitectureViewType.REQUEST_FLOW,
                system_prompt="""VIEW: REQUEST FLOW

OUTPUT CONTRACT (MANDATORY):
1) Output EXACTLY one Mermaid code block first, fenced as ```mermaid ... ```.
   - CRITICAL: NO CSS, NO @import, NO style tags, NO init blocks
   - ONLY pure Mermaid flowchart syntax (flowchart TD/LR)
   - ARROWS: Use --> for connections (NOT -> which is invalid)
2) Immediately after, output the layer explanations:

IMPORTANT (TITLE RULE): You MUST keep the layer numbers (Layer 1..5) but you MUST make the layer TITLES domain-specific to the given system.
Do NOT reuse generic titles like "Ingestion & Protection". Instead, name what this system actually does.
Example (for ride sharing): "### Layer 3 - Driver Matching & Geolocation".

### Layer 1 - [Entry Point / Ingress]
What happens:
- (2 bullets: entry point, initial validation)
Why it exists:
- (1 bullet: why this is needed)

### Layer 2 - [Gateway / Auth Layer]
What happens:
- (2 bullets: routing, authentication)
Why it exists:
- (1 bullet: architectural purpose)

### Layer 3 - [CORE BUSINESS LOGIC - Domain Specific]
How it works:
- (3-4 bullets with SPECIFIC mechanisms, algorithms, data structures)
- Include numbers: latency, throughput, sizes
- Show actual tech: "Redis GEORADIUS", not "caching layer"
- Example: "Scoring algorithm: 0.7*distance + 0.2*rating + 0.1*acceptance_rate"

Why this design:
- (1-2 bullets comparing alternatives with quantified reasoning)
- Example: "Chose Redis over PostGIS: 30ms vs 200ms latency at 100K QPS"

Key tradeoffs:
- (1-2 bullets on what you sacrificed for this design)
- Example: "Redis in-memory only → max 500K drivers (sufficient for 10-year growth)"

### Layer 4 - [Transaction / Persistence Layer]
How it works:
- (3-4 bullets with specific transaction handling, consistency guarantees)
- Include actual mechanisms: "Two-phase commit", "Optimistic locking", etc.

Why this design:
- (1-2 bullets explaining THE SPECIFIC CHOICE made here)

Key tradeoffs:
- (1-2 bullets on consistency vs availability vs latency)

### Layer 5 - [Post-Processing / Async Work]
How it works:
- (3-4 bullets on async processing, event handling)
- Show actual event flow and side effects

Why this design:
- (1-2 bullets on why async matters here)

Key tradeoffs:
- (1-2 bullets on eventual consistency or delayed feedback)

### Final End-to-End Flow Summary
(3-5 short sentences summarizing the complete flow)

Mermaid rules (CRITICAL):
- flowchart LR preferred
- ARROWS: A --> B (double dash arrow, NOT single ->)
- ASCII only
- <= 12 nodes and <= 15 edges
- No init blocks, no classDef, no linkStyle
- Short node labels (2-3 words)
""",
                user_prompt_template="""System: {system_description}

**CRITICAL**: Show a SPECIFIC, REAL user action for "{system_description}".
Use the actual terminology and workflows of THIS SYSTEM.
Think: "What is the MOST IMPORTANT user action in this system?"

Generate the critical end-to-end flow for THIS system.
""",
                constraints=[
                    "Maximum 20 nodes",
                    "Maximum 25 edges",
                    "Must include auth step",
                    "Must show database interaction",
                    "Linear flow (left to right)"
                ],
                max_nodes=20,
                max_edges=25
            ),
            
            ArchitectureViewType.ASYNC_PROCESSING: ViewGenerationPrompt(
                view_type=ArchitectureViewType.ASYNC_PROCESSING,
                system_prompt="""VIEW: ASYNC PROCESSING

OUTPUT CONTRACT (MANDATORY):
1) Output EXACTLY one Mermaid code block first, fenced as ```mermaid ... ```.
2) Immediately after, output the layer explanations:

IMPORTANT (TITLE RULE): You MUST keep the layer numbers (Layer 1..4) but you MUST make the layer TITLES domain-specific.
Do NOT reuse generic titles like "Event Publishing". Name what this system actually publishes.
Example (for e-commerce): "### Layer 1 - Order Event Stream".

### Layer 1 - [Event Publishing - Domain Specific]
How it works:
- (3-4 bullets with event schema, publish mechanism, ordering guarantees)
- Include: event format, topic/queue names, delivery semantics
- Example: "Kafka topic 'order-events', 10 partitions, at-least-once delivery"

Why this design:
- (1-2 bullets on why events vs synchronous calls)
- Example: "Chose Kafka over RabbitMQ: need message replay for exactly-once semantics"

Key tradeoffs:
- (1-2 bullets on eventual consistency cost, operational complexity)
- Example: "At-least-once delivery: consumers must be idempotent"

### Layer 2 - [Event Bus - Domain Specific]
How it works:
- (3-4 bullets with broker topology, partitioning, replication)
- Example: "Kafka cluster: 5 brokers, replication factor 3, min.insync.replicas=2"

Why this design:
- (1-2 bullets on durability vs throughput)

Key tradeoffs:
- (1-2 bullets on complexity vs reliability)

### Layer 3 - [Consumers - Domain Specific]
How it works:
- (3-4 bullets with consumer groups, parallelism, error handling)
- Example: "3 consumer groups in parallel: Inventory, Payment, Notification"

Why this design:
- (1-2 bullets on independent scaling)

Key tradeoffs:
- (1-2 bullets on message ordering vs parallelism)

### Layer 4 - [Side Effects - Domain Specific]
How it works:
- (3-4 bullets with actual side effects, idempotency, retries)
- Example: "Dead letter queue for failed events after 3 retry attempts"

Why this design:
- (1-2 bullets on failure handling strategy)

Key tradeoffs:
- (1-2 bullets on retry complexity vs reliability)

### Final Async Flow Summary
(3-5 short sentences summarizing async architecture)

Mermaid rules:
- flowchart TD or LR
- ASCII only
- <= 12 nodes and <= 15 edges
- No init blocks, no classDef, no linkStyle, no :::
- Avoid subgraphs unless absolutely necessary
- Short node labels (2-3 words)
""",
                user_prompt_template="""System: {system_description}

**CRITICAL**: Show async patterns SPECIFIC to "{system_description}".
Think about: What heavy work happens in the background? What needs eventual consistency?
Use the actual background jobs and async workflows of THIS SYSTEM.

Generate ASYNC PROCESSING architecture with:

1. Mermaid diagram (event bus + workers doing THIS SYSTEM's actual background jobs)
2. Layer-by-layer breakdown (what async work THIS SYSTEM actually does)
3. Each layer: What happens IN THIS SYSTEM + Why async matters HERE
4. Final async flow summary (how THIS SYSTEM achieves eventual consistency)

Focus on what happens AFTER an event is published IN THIS SPECIFIC SYSTEM.""",
                constraints=[
                    "Maximum 12 nodes",
                    "Maximum 15 edges",
                    "Must show event bus",
                    "Must show at least 2 workers",
                    "Top-down flow"
                ],
                max_nodes=12,
                max_edges=15
            ),
            
            ArchitectureViewType.DATA_MODEL: ViewGenerationPrompt(
                view_type=ArchitectureViewType.DATA_MODEL,
                system_prompt="""VIEW: DATA & STORAGE

OUTPUT CONTRACT (MANDATORY):
1) Output EXACTLY one Mermaid code block first, fenced as ```mermaid ... ```.
   - CRITICAL: NO CSS, NO @import, NO style tags, NO init blocks
   - ONLY pure Mermaid flowchart syntax (flowchart TD/LR)
   - ARROWS: Use --> for connections (NOT -> which is invalid)
2) Immediately after, output the layer explanations:

IMPORTANT (TITLE RULE): You MUST keep the layer numbers (Layer 1..3) but you MUST make the layer TITLES domain-specific.
Do NOT reuse generic titles like "Primary Storage". Name the real entities/stores for this system (e.g., "User Profile Store", "Catalog Index", "Session Cache").

### Layer 1 - [Primary Storage - Domain Specific]
How it works:
- (3-4 bullets with SPECIFIC storage tech, schema design, partitioning)
- Include: database type, indexing strategy, sharding approach
- Example: "User profiles in PostgreSQL, partitioned by user_id % 128"

Why this design:
- (1-2 bullets comparing alternatives: SQL vs NoSQL vs NewSQL)
- Example: "Chose Postgres over Cassandra: strong consistency for user profiles"

Key tradeoffs:
- (1-2 bullets on consistency, scalability, cost)
- Example: "Vertical scaling limit: 10TB per shard, max 128 shards = 1.28PB"

### Layer 2 - [Caching & Speed - Domain Specific]
How it works:
- (3-4 bullets with cache topology, TTL, eviction policy)
- Example: "Redis cluster: 3 primary + 3 replica, LRU eviction, 1-hour TTL"

Why this design:
- (1-2 bullets on cache hit rate, latency improvement)
- Example: "95% cache hit rate → 200ms → 5ms read latency"

Key tradeoffs:
- (1-2 bullets on staleness vs freshness)
- Example: "Cache invalidation lag: up to 10 seconds stale data acceptable"

### Layer 3 - [Search & Analytics - Domain Specific]
How it works:
- (3-4 bullets with search engine, indexing, query optimization)
- Example: "Elasticsearch: inverted index rebuilt every 60s, 5-shard cluster"

Why this design:
- (1-2 bullets on search requirements vs batch analytics)

Key tradeoffs:
- (1-2 bullets on query latency vs index freshness)

### Final Data Flow Summary
(3-5 sentences summarizing data patterns)

Mermaid rules (CRITICAL):
- flowchart LR
- ARROWS: A --> B (double dash arrow, NOT single ->)
- ASCII only
- <= 10 nodes and <= 14 edges
- No init blocks, no classDef, no linkStyle
- Short node labels
- Use cylinder syntax for data stores when applicable: DB[(Name)]
- Avoid subgraphs unless absolutely necessary
""",
                user_prompt_template="""System: {system_description}

**CRITICAL**: Show data patterns SPECIFIC to "{system_description}".
Think about: What TYPE of data does this system store? How is it partitioned? What needs to be fast?
Use the actual data entities and access patterns of THIS SYSTEM.

Generate a compact data view for THIS system.
""",
                constraints=[
                    "Maximum 10 data stores",
                    "Maximum 15 edges",
                    "Must use cylinder shape for databases: [(Name)]",
                    "Include technology names"
                ],
                max_nodes=10,
                max_edges=15
            ),
            
            ArchitectureViewType.DEPLOYMENT: ViewGenerationPrompt(
                view_type=ArchitectureViewType.DEPLOYMENT,
                system_prompt="""VIEW: DEPLOYMENT ARCHITECTURE

OUTPUT CONTRACT (MANDATORY):
1) Output EXACTLY one Mermaid code block first, fenced as ```mermaid ... ```.
   - CRITICAL: NO CSS, NO @import, NO style tags, NO init blocks
   - ONLY pure Mermaid flowchart syntax (flowchart TD/LR)
   - ARROWS: Use --> for connections (NOT -> which is invalid)
2) Immediately after, output the layer explanations:

IMPORTANT (TITLE RULE): You MUST keep the layer numbers (Layer 1..5) but you MUST make the layer TITLES domain-specific.
Do NOT reuse generic titles like "Edge & Traffic". Name the real infra concerns for this system (e.g., "Global Routing & CDN", "API Edge Auth", "Service Mesh").

### Layer 1 - [Edge & CDN - Domain Specific]
How it works:
- (3-4 bullets with CDN setup, routing, DDoS mitigation, global distribution)
- Include: PoP count, latency targets, caching strategy
- Example: "CloudFlare CDN: 200+ PoPs, 50ms p95 latency globally, 95% cache hit rate"

Why this design:
- (1-2 bullets on latency requirements, cost savings from caching)
- Example: "CDN reduces origin load by 85% → $30K/month savings on compute"

Key tradeoffs:
- (1-2 bullets on cost vs performance, cache invalidation complexity)
- Example: "CDN costs $5K/month but saves $30K in origin compute + bandwidth"

### Layer 2 - [API Gateway & Governance - Domain Specific]
How it works:
- (3-4 bullets with gateway topology, rate limiting, auth validation)
- Example: "Kong API Gateway: 3 replicas, Redis-backed rate limiter (1000 req/min per user)"

Why this design:
- (1-2 bullets on centralized control, security)

Key tradeoffs:
- (1-2 bullets on single point of failure mitigation, latency overhead)

### Layer 3 - [Core Service Deployment - Domain Specific]
How it works:
- (3-4 bullets with orchestration, scaling, service mesh)
- Example: "Kubernetes: 20 pods per service, HPA scales 10-100 pods at 70% CPU"

Why this design:
- (1-2 bullets on deployment strategy: blue-green, canary, rolling)
- Example: "Canary deploys: 5% traffic → 50% → 100% over 30 minutes"

Key tradeoffs:
- (1-2 bullets on cost vs resilience)
- Example: "Over-provisioned 30% for traffic spikes → $15K/month extra cost"

### Layer 4 - [Async Workers & Background Jobs - Domain Specific]
How it works:
- (3-4 bullets with job queues, workers, retry logic, scaling)
- Example: "SQS FIFO queues, 10 worker instances, exponential backoff retry (1s → 30s → 5min)"

Why this design:
- (1-2 bullets on decoupling, fault tolerance)
- Example: "Workers scale independently: peak 100 workers, off-peak 5 workers"

Key tradeoffs:
- (1-2 bullets on latency vs reliability, cost of retries)
- Example: "Async processing adds 5-30s latency but prevents API timeouts"

### Layer 5 - [Data Persistence & Backups - Domain Specific]
How it works:
- (3-4 bullets with backup strategy, replication, disaster recovery, RPO/RTO)
- Example: "RDS Multi-AZ, hourly snapshots, cross-region replication (us-east-1 → us-west-2)"

Why this design:
- (1-2 bullets on RTO/RPO requirements, compliance)
- Example: "RPO=1 hour, RTO=15 minutes for 99.99% availability SLA"

Key tradeoffs:
- (1-2 bullets on cost vs availability, storage costs)
- Example: "Cross-region replication adds $2K/month but meets disaster recovery SLA"

### Final Deployment Summary
(3-5 sentences summarizing deployment architecture)

Mermaid rules (CRITICAL):
- flowchart TD preferred (top-down for infrastructure)
- ARROWS: A --> B (double dash arrow, NOT single ->)
- ASCII only
- <= 12 nodes and <= 15 edges
- No init blocks, no classDef, no linkStyle
- Short node labels
""",
                user_prompt_template="""System: {system_description}

**CRITICAL**: Show infrastructure SPECIFIC to the requirements of "{system_description}".
Think about: What scale? What latency needs? What geographic distribution?
Explain infrastructure choices based on the actual needs of THIS SYSTEM.

Generate a compact deployment view for THIS system.
""",
                constraints=[
                    "Maximum 12 nodes",
                    "Maximum 15 edges",
                    "Use subgraphs for network boundaries",
                    "Show replica counts"
                ],
                max_nodes=12,
                max_edges=15
            ),
            
            ArchitectureViewType.OBSERVABILITY: ViewGenerationPrompt(
                view_type=ArchitectureViewType.OBSERVABILITY,
                system_prompt="""VIEW: OBSERVABILITY & MONITORING

OUTPUT CONTRACT (MANDATORY):
1) Output EXACTLY one Mermaid code block first, fenced as ```mermaid ... ```.
   - CRITICAL: NO CSS, NO @import, NO style tags, NO init blocks
   - ONLY pure Mermaid flowchart syntax (flowchart TD/LR)
   - ARROWS: Use --> for data flow, -.-> for monitoring (dotted arrow)
2) Immediately after, output the layer explanations:

IMPORTANT (TITLE RULE): You MUST keep the layer numbers (Layer 1..4) but you MUST make the layer TITLES domain-specific.
Do NOT reuse generic titles like "Metrics". Name what you actually measure/log/trace/alert on for this system (e.g., "Checkout Latency SLOs", "Fraud Signals", "Ranking Drift").

Use "Problem → Solution → Impact" structure for observability layers:

### Layer 1 - [Metrics - Domain Specific]
Problem this layer solves:
- (1-2 bullets on what BREAKS without metrics)
- Example: "Without metrics: Cannot detect 200ms → 2s latency regression"

How we solve it:
- (2-3 bullets with actual metrics, tools, dashboards)
- Example: "Prometheus scrapes /metrics every 15s, Grafana dashboards, 4 golden signals"

Measurable impact:
- (1-2 bullets with NUMBERS)
- Example: "Detected payment gateway timeout in 90 seconds vs 45 minutes before"

### Layer 2 - [Logs - Domain Specific]
Problem this layer solves:
- (1-2 bullets on debugging challenges)

How we solve it:
- (2-3 bullets with log aggregation, structured logging)
- Example: "ELK stack, JSON logs, request_id correlation, 7-day retention"

Measurable impact:
- (1-2 bullets on MTTR improvement)

### Layer 3 - [Traces - Domain Specific]
Problem this layer solves:
- (1-2 bullets on distributed system debugging)
- Example: "15-microservice call chain: which service is slow?"

How we solve it:
- (2-3 bullets with distributed tracing)
- Example: "Jaeger spans, trace_id propagation, waterfall view"

Measurable impact:
- (1-2 bullets on root cause time)
- Example: "MTTR: 45 min → 8 min to find failing service"

### Layer 4 - [Alerts - Domain Specific]
Problem this layer solves:
- (1-2 bullets on incident response)

How we solve it:
- (2-3 bullets with alerting rules, on-call rotation)
- Example: "PagerDuty: p50 latency > 100ms for 5 min → page on-call"

Measurable impact:
- (1-2 bullets on incident detection speed)

### Final Observability Summary
(3-5 sentences summarizing monitoring strategy)

Mermaid rules (CRITICAL):
- flowchart LR
- ARROWS: A --> B for data flow, A -.-> B for monitoring
- ASCII only
- <= 10 nodes and <= 12 edges
- No init blocks, no classDef, no linkStyle
- Short node labels
""",
                user_prompt_template="""System: {system_description}

**CRITICAL**: Everything must be SPECIFIC to "{system_description}".
Think about: What are the critical failures? What metrics matter most? What code pattern is most important?
Monitor what matters for THIS SYSTEM. Implement code that solves THIS SYSTEM's core challenge.

Generate a compact observability view for THIS system.
""",
                constraints=[
                    "Maximum 10 nodes",
                    "Maximum 12 edges",
                    "Must show metrics + logs + alerts",
                    "Left to right flow"
                ],
                max_nodes=10,
                max_edges=12
            ),
            
            ArchitectureViewType.SECURITY: ViewGenerationPrompt(
                view_type=ArchitectureViewType.SECURITY,
                system_prompt="""You are a security architect documenting SECURITY ARCHITECTURE.

CRITICAL RULES:
1. Show authentication flow (OAuth, JWT, etc.)
2. Show authorization layers (RBAC, policies)
3. Show security boundaries (firewalls, WAF)
4. Show encryption (TLS, at-rest)
5. Maximum 10 components

This diagram is for security teams and compliance.
It must show how the system is protected.""",
                user_prompt_template="""Create a security architecture diagram for: {system_description}

Generate a Mermaid flowchart (flowchart TD) showing:
- Authentication (OAuth, JWT, SSO)
- Authorization (RBAC, policies)
- Security boundaries (WAF, firewall)
- Encryption (TLS, KMS)
- Secrets management (Vault, AWS Secrets)

Output ONLY the Mermaid code, starting with 'flowchart TD'.""",
                constraints=[
                    "Maximum 10 nodes",
                    "Maximum 12 edges",
                    "Must show auth + authz",
                    "Show encryption points"
                ],
                max_nodes=10,
                max_edges=12
            )
        }
    
    def get_recommended_views(self, system_description: str, user_level: str = "mid") -> List[ArchitectureViewType]:
        """
        Determine which views to generate based on system description and user level.
        
        Args:
            system_description: Description of the system to design
            user_level: junior|mid|senior|architect
            
        Returns:
            List of view types to generate
        """
        # Core views everyone needs
        core_views = [
            ArchitectureViewType.SYSTEM_OVERVIEW,
            ArchitectureViewType.REQUEST_FLOW,
        ]
        
        # Detect if system has async processing
        async_keywords = ["event", "queue", "async", "worker", "background", "kafka", "rabbitmq", "sqs"]
        has_async = any(keyword in system_description.lower() for keyword in async_keywords)
        
        # Detect if system has complex data needs
        data_keywords = ["database", "cache", "redis", "postgres", "mongodb", "elasticsearch", "search"]
        has_data = any(keyword in system_description.lower() for keyword in data_keywords)
        
        # Build view list based on complexity
        views = core_views.copy()
        
        if has_async:
            views.append(ArchitectureViewType.ASYNC_PROCESSING)
        
        if has_data or user_level in ["senior", "architect"]:
            views.append(ArchitectureViewType.DATA_MODEL)
        
        if user_level in ["senior", "architect"]:
            views.extend([
                ArchitectureViewType.DEPLOYMENT,
                ArchitectureViewType.OBSERVABILITY
            ])
        
        # Security is optional but recommended for production systems
        security_keywords = ["auth", "security", "oauth", "jwt", "rbac", "permission"]
        if any(keyword in system_description.lower() for keyword in security_keywords):
            views.append(ArchitectureViewType.SECURITY)
        
        return views
    
    def get_view_prompt(self, view_type: ArchitectureViewType, system_description: str) -> Dict[str, str]:
        """
        Get the system and user prompts for a specific view.
        
        Args:
            view_type: Type of view to generate
            system_description: Description of the system
            
        Returns:
            Dict with 'system_prompt' and 'user_prompt'
        """
        prompt_config = self.view_prompts.get(view_type)
        if not prompt_config:
            raise ValueError(f"Unknown view type: {view_type}")
        
        user_prompt = prompt_config.user_prompt_template.format(
            system_description=system_description
        )
        
        max_nodes, max_edges = self._dynamic_limits(view_type, system_description)

        # Append dynamic limits to the prompt so sizing is driven by requirements.
        # We keep the original per-view guidance, but override numeric caps here.
        system_prompt = (
            (prompt_config.system_prompt or "").rstrip()
            + "\n\nDynamic size limits for THIS question (must obey):\n"
            + f"- <= {max_nodes} nodes\n"
            + f"- <= {max_edges} edges\n"
        )

        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "constraints": prompt_config.constraints,
            "max_nodes": max_nodes,
            "max_edges": max_edges,
        }
    
    def get_view_metadata(self, view_type: ArchitectureViewType) -> Dict[str, Any]:
        """Get metadata about a specific view type."""
        
        metadata = {
            ArchitectureViewType.SYSTEM_OVERVIEW: {
                "title": "🏗️ System Overview",
                "description": "High-level building blocks and major components",
                "complexity_level": "junior",
                "estimated_explanation_time": "1-2 min",
                "audience": "Everyone (executives, PMs, engineers)",
                "key_question": "What are the major building blocks?"
            },
            ArchitectureViewType.REQUEST_FLOW: {
                "title": "🔁 Request Flow",
                "description": "Critical business path during user action",
                "complexity_level": "mid",
                "estimated_explanation_time": "2-3 min",
                "audience": "Backend engineers, interviewers",
                "key_question": "What happens when a user performs an action?"
            },
            ArchitectureViewType.ASYNC_PROCESSING: {
                "title": "⚙️ Async & Event Processing",
                "description": "Background processing and event-driven workflows",
                "complexity_level": "mid",
                "estimated_explanation_time": "2-3 min",
                "audience": "Backend engineers, distributed systems engineers",
                "key_question": "How does the system handle background work?"
            },
            ArchitectureViewType.DATA_MODEL: {
                "title": "🗄️ Data & Storage Model",
                "description": "Where data lives and persistence strategy",
                "complexity_level": "mid",
                "estimated_explanation_time": "2-3 min",
                "audience": "Backend engineers, database engineers",
                "key_question": "Where does data live and why?"
            },
            ArchitectureViewType.DEPLOYMENT: {
                "title": "📦 Deployment Architecture",
                "description": "Infrastructure, scaling, and cloud services",
                "complexity_level": "senior",
                "estimated_explanation_time": "3-4 min",
                "audience": "DevOps, SRE, infrastructure engineers",
                "key_question": "How is the system deployed and scaled?"
            },
            ArchitectureViewType.OBSERVABILITY: {
                "title": "📊 Observability & Reliability",
                "description": "Monitoring, logging, and operational health",
                "complexity_level": "senior",
                "estimated_explanation_time": "2-3 min",
                "audience": "SRE, operations, on-call engineers",
                "key_question": "How do we monitor and keep it healthy?"
            },
            ArchitectureViewType.SECURITY: {
                "title": "🛡️ Security Architecture",
                "description": "Authentication, authorization, and protection",
                "complexity_level": "senior",
                "estimated_explanation_time": "3-4 min",
                "audience": "Security engineers, compliance teams",
                "key_question": "How is the system protected?"
            }
        }
        
        return metadata.get(view_type, {})
    
    def validate_diagram_complexity(
        self,
        mermaid_code: str,
        view_type: ArchitectureViewType,
        max_nodes: int | None = None,
        max_edges: int | None = None,
    ) -> Dict[str, Any]:
        """
        Validate that a generated diagram meets complexity constraints.
        
        Returns:
            Dict with 'valid' (bool) and 'issues' (list of strings)
        """
        import re
        
        prompt_config = self.view_prompts.get(view_type)
        if not prompt_config:
            return {"valid": True, "issues": []}
        
        issues = []
        
        # Count nodes (lines with node definitions)
        node_pattern = re.compile(r'^\s*\w+[\[\(\{]', re.MULTILINE)
        node_count = len(node_pattern.findall(mermaid_code))
        
        # Count edges (lines with arrows)
        edge_pattern = re.compile(r'--+>|=+>|~+>|-\.+->', re.MULTILINE)
        edge_count = len(edge_pattern.findall(mermaid_code))
        
        limit_nodes = int(max_nodes) if max_nodes is not None else int(prompt_config.max_nodes)
        limit_edges = int(max_edges) if max_edges is not None else int(prompt_config.max_edges)

        if node_count > limit_nodes:
            issues.append(f"Too many nodes: {node_count} (max: {limit_nodes})")
        
        if edge_count > limit_edges:
            issues.append(f"Too many edges: {edge_count} (max: {limit_edges})")
        
        # Check for subgraphs in views that shouldn't have them
        if view_type == ArchitectureViewType.SYSTEM_OVERVIEW:
            if "subgraph" in mermaid_code.lower():
                issues.append("System Overview should not use subgraphs (too complex)")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "node_count": node_count,
            "edge_count": edge_count,
            "max_nodes": limit_nodes,
            "max_edges": limit_edges,
        }


# Singleton instance
_architecture_generator = None

def get_architecture_generator() -> ArchitectureGeneratorService:
    """Get singleton instance of architecture generator."""
    global _architecture_generator
    if _architecture_generator is None:
        _architecture_generator = ArchitectureGeneratorService()
    return _architecture_generator
