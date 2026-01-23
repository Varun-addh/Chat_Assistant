from __future__ import annotations

import re

from app.config import settings


_SYSTEM_DESIGN_EXTRAS: tuple[str, ...] = (
	# Product-like prompts / canonical system design interview topics
	"url shortener",
	"tinyurl",
	"websocket",
	"sse",
	"fanout",
	"feed",
	# Common infra building blocks
	"api gateway",
	"rate limiter",
	"cdn",
)


def _system_design_keywords() -> tuple[str, ...]:
	"""Single source of truth for system design detection keywords.

	Uses settings-based architecture detection signals as the base, then adds a
	few extra canonical prompts. This avoids duplicated hardcoded lists spread
	across modules.
	"""
	base: list[str] = []
	base.extend(getattr(settings, "architecture_detection_explicit_keywords", []) or [])
	base.extend(getattr(settings, "architecture_detection_system_concepts_scale", []) or [])
	base.extend(getattr(settings, "architecture_detection_system_concepts_data", []) or [])
	base.extend(getattr(settings, "architecture_detection_system_concepts_infra", []) or [])
	# Normalize and de-dup while preserving order
	seen: set[str] = set()
	out: list[str] = []
	for k in [*(base or []), *_SYSTEM_DESIGN_EXTRAS]:
		k = (k or "").strip().lower()
		if not k or k in seen:
			continue
		seen.add(k)
		out.append(k)
	return tuple(out)

_DATABASE_SCHEMA_HINTS: tuple[str, ...] = (
	"database schema",
	"er diagram",
	"entity relationship",
	"schema design",
	"table design",
	"relational model",
)

_UI_HINTS: tuple[str, ...] = (
	"wireframe",
	"ui design",
	"ux",
	"user interface",
	"frontend design",
	"mockup",
)

_ALGORITHM_HINTS: tuple[str, ...] = (
	"algorithm",
	"data structure",
	"leetcode",
	"time complexity",
	"big o",
	"dp",
	"dynamic programming",
)


def is_system_design_question(question: str) -> bool:
	"""Heuristic classifier for System Design / Architecture questions.

	Kept intentionally conservative because `LLMService` checks this intent before
	database-schema/UI/algorithm routes.
	"""
	q = (question or "").strip().lower()
	if not q:
		return False

	# Avoid stealing queries that are clearly a different diagram intent.
	if any(h in q for h in _DATABASE_SCHEMA_HINTS):
		return False
	if any(h in q for h in _UI_HINTS):
		return False

	# Direct system-design signals.
	if any(k in q for k in _system_design_keywords()):
		return True

	# If it looks purely algorithmic and has no architecture terms, do not route to system design.
	if any(h in q for h in _ALGORITHM_HINTS):
		return False

	# Common phrasing patterns.
	if re.search(r"\bdesign\s+(a|an|the)?\s*(?:\w+\s+){0,4}(system|service|platform|application|app)\b", q):
		return True
	if re.search(r"\bhow\s+would\s+you\s+(?:design|architect)\b", q):
		return True

	return False


def greeting_overrides() -> str:
	return (
		"\n\nGreeting Overrides (apply only to salutations/thanks/parting):\n"
		"- Do NOT start with any 'Complete Answer' bullets or a Summary.\n"
		"- No headings. Respond briefly (one or two sentences) in a friendly tone.\n"
		"- Acknowledge the greeting/thanks and offer help if appropriate.\n"
		"- Do NOT output onboarding sections like 'Introduction', 'How I Can Assist You', 'Getting Started', or 'Example Questions'.\n"
		"- If the user shares their name, address them by name and ask what they'd like to practice.\n"
	)


def off_topic_overrides() -> str:
	return (
		"\n\nOff-Topic Query Overrides (apply only to non-interview questions):\n"
		"- Politely redirect to interview preparation topics.\n"
		"- Format: 'That's an interesting question, but let's focus on interview preparation. Would you like help with [relevant topic]?'\n"
		"- Suggest relevant interview topics: technical concepts, coding problems, system design, behavioral questions.\n"
		"- Keep response brief and professional.\n"
	)


def ambiguous_query_overrides() -> str:
	return (
		"\n\nAmbiguous Query Overrides (apply only to unclear questions):\n"
		"- Ask 1-2 specific clarifying questions before proceeding.\n"
		"- Format: 'Could you clarify what specific aspect of [topic] you'd like to discuss?'\n"
		"- Provide examples of what you could help with.\n"
		"- Keep response brief and helpful.\n"
	)


def context_fallback_overrides() -> str:
	return (
		"\n\nContext Fallback Overrides (apply when context is insufficient):\n"
		"- If no past context available: Proceed with fresh, standalone answer.\n"
		"- If context is insufficient: Acknowledge and provide comprehensive answer.\n"
		"- For pronouns without clear referents: Ask for clarification or provide general answer.\n"
		"- When context is unclear: 'Based on general interview practices...'\n"
		"- Always ensure answers work independently of conversation history.\n"
	)


def comparison_overrides(question: str) -> str:
	_ = question
	return (
		"\n\nComparison Format Overrides (apply only to comparison questions):\n"
		"- Produce ONE concise markdown table with headers: | Feature | A | B |.\n"
		"- Use clear, compact rows such as Definition, Core Function, Input, Output, Autonomy, Examples, Use Case Focus, Decision Making.\n"
		"- Keep cells short (1–2 lines).\n"
		"- After the table, add an 'In short:' section with 2 bullet points summarizing A vs B in one sentence each.\n"
		"- No extra headings, no duplicate sections, no verbose paragraphs.\n"
	)


def database_schema_overrides() -> str:
	"""Overrides for database schema questions"""
	return (
		"\n\nDatabase Schema Overrides (apply only to database schema questions):\n"
		"- Include a 'Database Schema' section with an ER diagram using Mermaid.\n"
		"- Use erDiagram syntax with entities, relationships, and attributes.\n"
		"- Example format:\n"
		"  ```mermaid\n"
		"  erDiagram\n"
		"    USER ||--o{ ORDER : places\n"
		"    USER {\n"
		"      int id PK\n"
		"      string name\n"
		"      string email\n"
		"    }\n"
		"    ORDER {\n"
		"      int id PK\n"
		"      int user_id FK\n"
		"      decimal total\n"
		"    }\n"
		"  ```\n"
	)


def ui_design_overrides() -> str:
	"""Overrides for UI design questions"""
	return (
		"\n\nUI Design Overrides (apply only to UI/UX design questions):\n"
		"- Include a 'UI Design' section with a wireframe or layout diagram using Mermaid.\n"
		"- Use flowchart syntax to show component hierarchy and layout.\n"
		"- Example format:\n"
		"  ```mermaid\n"
		"  flowchart TD\n"
		"    A[Header] --> B[Navigation]\n"
		"    A --> C[Search Bar]\n"
		"    A --> D[User Menu]\n"
		"    E[Main Content] --> F[Article List]\n"
		"    E --> G[Sidebar]\n"
		"    H[Footer] --> I[Links]\n"
		"  ```\n"
	)


def algorithm_overrides() -> str:
	"""Overrides for algorithm questions"""
	return (
		"\n\nAlgorithm Overrides (apply only to algorithm questions):\n"
		"- Include a 'Algorithm Flow' section with a flowchart using Mermaid.\n"
		"- Use flowchart syntax to show the algorithm steps and decision points.\n"
		"- Example format:\n"
		"  ```mermaid\n"
		"  flowchart TD\n"
		"    A[Start] --> B{Input Valid?}\n"
		"    B -->|Yes| C[Process Data]\n"
		"    B -->|No| D[Return Error]\n"
		"    C --> E[Return Result]\n"
		"  ```\n"
	)


def system_design_overrides() -> str:
	"""Enforce the System Design response structure requested by the user."""
	return (
		"\n\nSystem Design Overrides (apply only to system/architecture questions):\n"
		"- Follow this exact markdown structure:\n"
		"\n### **Key Highlights**\n"
		"- 4–6 crisp bullets on core data structures, pipelines, algorithms, scalability ideas, trade-offs.\n"
		"\n### **Detailed Explanation**\n"
		"\n#### **1. Requirements Analysis**\n"
		"- **Functional Requirements:** Core outcomes.\n"
		"- **Non-Functional Requirements:** Latency/availability/scalability/freshness.\n"
		"\n#### **2. High-Level Architecture**\n"
		"- Provide a table with Component | Purpose | Technology/Layer.\n"
		"- Executive summary (copy-pasteable): Summarize the domain-specific strategy in 2–4 sentences. Example patterns to consider and adapt: streaming pipelines, event-driven fanout, CQRS, serverless ingestion, microservices vs monolith, or OLAP/OLTP separation. Choose stacks per domain and scale (e.g., messaging vs media vs ridesharing), and justify key trade-offs briefly.\n"
		"- **MANDATORY: Include a 'Visual Architecture Diagram' section with a Mermaid flowchart code block.**\n"
		"- **ALWAYS generate at least one domain-relevant Mermaid diagram (system, data, or cloud view depending on the question), not optional.**\n"
		"- **Generate diagrams for ALL architecture questions: system design, cloud architecture, data architecture, security architecture, etc.**\n"
		"\n"
		"- **🎯 COMPLEXITY REQUIREMENT - THIS IS MANDATORY, NOT OPTIONAL:**\n"
		"  **Generate PRODUCTION-GRADE, FAANG-INTERVIEW-LEVEL architectures with:**\n"
		"  * **Minimum 20-30 components** for comprehensive real-world systems (NOT 6-10 simple boxes!)\n"
		"  * **8-15 microservices** specific to the domain (NOT generic 'Backend Service')\n"
		"  * **Multiple layers:** Client → Edge (CDN/LB) → Gateway (API GW/Auth) → Services (8-15 services) → Message Queue → Workers → Data (multiple DBs) → Monitoring\n"
		"  * **Event-driven architecture:** Kafka/RabbitMQ for async communication between services\n"
		"  * **Background processing:** Workers, job queues, schedulers, processors\n"
		"  * **Multiple data stores:** Primary DB, Read Replicas, Cache (Redis), Search (Elasticsearch), Time-series, Graph DB as needed\n"
		"  * **Observability stack:** Metrics (Prometheus), Logs (ELK), Traces (Jaeger), Alerts\n"
		"  * **Domain logic:** Ranking, Recommendation, ML inference, real-time processing as needed\n"
		"\n"
		"- **🚨 REJECTION CRITERIA - Your design is INSUFFICIENT if it has:**\n"
		"  * Fewer than 15 distinct components\n"
		"  * Only 2-4 generic services (like 'User Service' and 'Backend Service')\n"
		"  * No message queue / event streaming\n"
		"  * No background workers or async processing\n"
		"  * Only one database\n"
		"  * No caching strategy beyond basic cache\n"
		"  * No monitoring/observability components\n"
		"  * No domain-specific services\n"
		"\n"
		"- **📐 DIAGRAM SCALE GUIDELINES:**\n"
		"  * **ALWAYS DEFAULT TO LARGE/ENTERPRISE SCALE**\n"
		"  * MVP/Simple: Only if user EXPLICITLY asks for 'simple', 'basic', or 'MVP'\n"
		"  * Standard: 20-25 components with full microservices decomposition\n"
		"  * Enterprise: 25-35 components with observability, ML, and advanced patterns\n"
		"\n"
		"- **🧠 DYNAMIC DOMAIN ANALYSIS - For EVERY system design question:**\n"
		"  1. **Identify the core domain:** What problem is being solved? (social, commerce, messaging, streaming, logistics, fintech, etc.)\n"
		"  2. **Determine core workflows:** What are the main user journeys and data flows?\n"
		"  3. **Identify domain-specific services:** What microservices are ESSENTIAL for this specific domain?\n"
		"     - Ask yourself: 'What would a FAANG company build for this system?'\n"
		"     - Think: 'What services does Twitter/Instagram/Uber/Netflix/Amazon actually have?'\n"
		"  4. **Determine data requirements:** What types of data stores does this domain need?\n"
		"     - Relational for transactions? Graph for relationships? Time-series for metrics? Search for queries?\n"
		"  5. **Identify async workflows:** What operations should be event-driven or background processed?\n"
		"  6. **Add supporting infrastructure:** Monitoring, logging, alerting, rate limiting, circuit breakers\n"
		"\n"
		"- **🔍 BEFORE generating any architecture, mentally decompose the system:**\n"
		"  * What are the 3-5 core user actions? → Each needs dedicated services\n"
		"  * What data needs to be stored? → Multiple specialized data stores\n"
		"  * What needs real-time processing? → WebSockets, streaming, push notifications\n"
		"  * What can be async? → Message queues, workers, batch processing\n"
		"  * What needs ML/ranking? → Recommendation, personalization, fraud detection\n"
		"  * What external integrations exist? → Payment, email, SMS, maps, CDN\n"
		"\n"
		"- Use solid arrows (-->), subgraphs for layers (User, Backend, Services, Cache, Database), and colorful classDefs.\n"
		"- Choose appropriate flowchart direction: TD (top-down) for layered architectures, LR (left-right) for data flow.\n"
		"- Include all major components: clients, load balancers, API gateways, microservices, databases, caches, message queues.\n"
		"- Use descriptive node names and proper styling with classDef statements.\n"
		"- Adapt the diagram to the specific architecture type (system, cloud, data, security, etc.).\n"
		"- **CRITICAL: Only include components that are directly connected in the data flow. NO floating or disconnected nodes.**\n"
		"- **NO conceptual layers or standalone legend boxes. Every node must have incoming/outgoing connections.**\n"
		"- **Focus on functional components that actively participate in request/response flows.**\n"
		"\n"
		"- **🔢 NUMBERED FLOW SEQUENCE (MANDATORY):**\n"
		"  * **EVERY arrow/edge MUST have a step number to show the sequence of operations**\n"
		"  * Format: `NodeA -->|1. Action| NodeB` or `NodeA -->|2. Process| NodeC`\n"
		"  * Start numbering from 1 at the entry point (usually Client/User)\n"
		"  * Follow the logical request flow: Client → Edge → Gateway → Services → Data\n"
		"  * Show response flow with higher numbers: Data → Services → Client\n"
		"  * Example numbered edges:\n"
		"    - `Client -->|1. Request| CDN`\n"
		"    - `CDN -->|2. Forward| LB`\n"
		"    - `LB -->|3. Route| APIGateway`\n"
		"    - `APIGateway -->|4. Auth| AuthService`\n"
		"    - `AuthService -->|5. Validate| UserDB`\n"
		"  * **This helps users understand the exact flow sequence!**\n"
		"\n"
		"- **🔗 CONNECTION REQUIREMENTS (MANDATORY):**\n"
		"  * **EVERY subgraph must be connected to other subgraphs via at least one edge**\n"
		"  * **NO isolated/floating subgraphs allowed** - Monitoring, Analytics, Workers all need connections\n"
		"  * Connect Monitoring to Services: `Services -->|metrics| Prometheus`\n"
		"  * Connect Workers to Queues: `Kafka -->|consume| Workers`\n"
		"  * Connect Analytics to Data: `Spark -->|read| DataLake`\n"
		"  * **If a component exists in the diagram, it MUST participate in the data flow**\n"
		"\n"
		"- **CRITICAL Mermaid syntax rules for beautiful, seamless UX:**\n"
		"  * Use simple node IDs (no spaces, special chars, or hyphens)\n"
		"  * Keep subgraph names simple (no spaces, use underscores)\n"
		"  * Every classDef must be applied to nodes using ::: syntax\n"
		"  * Avoid complex edge labels - use simple arrows (-->) or basic labels\n"
		"  * Test syntax: ensure all node references match defined nodes\n"
		"  * **NO COMMENTS** - Mermaid doesn't support % comments in diagrams\n"
		"  * **NO SPECIAL CHARACTERS** in edge labels - use plain arrows only\n"
		"  * **NO NUMBERS** in edge labels - they cause parse errors\n"
		"  * **SUBGRAPH SYNTAX:** Use `subgraph ID[Title]` format, not `subgraph Title[Title]`\n"
		"  * **NO FLOATING NODES:** Every node must be inside a subgraph or connected to the main flow\n"
		"  * **CONSISTENT RENDERING:** Avoid complex subgraph nesting that causes renderer differences\n"
		"- **Visual Design Excellence:**\n"
		"  * Use consistent, professional color palette with high contrast\n"
		"  * Apply rounded rectangles for modern look: use [Node Name] not (Node Name)\n"
		"  * Create visual hierarchy with subgraphs and proper spacing\n"
		"  * Use meaningful, concise node labels (max 2-3 words)\n"
		"  * Organize flow logically: top-to-bottom for layered, left-to-right for sequential\n"
		"  * Apply consistent styling: same node types get same colors\n"
		"  * Use database icons for storage: [(Database)] syntax\n"
		"  * Use cloud icons for external services: [(Cloud Service)] syntax\n"
		"- **Professional Color Scheme:**\n"
		"  * Client: Light blue (#e3f2fd) with dark blue stroke (#1976d2)\n"
		"  * Edge/Gateway: Light purple (#f3e5f5) with purple stroke (#7b1fa2)\n"
		"  * Services: Light orange (#fff3e0) with orange stroke (#f57c00)\n"
		"  * Data/Storage: Light green (#e8f5e9) with green stroke (#388e3c)\n"
		"  * Queue/Cache: Light yellow (#fffde7) with amber stroke (#f9a825)\n"
		"- **MANDATORY: After the Mermaid diagram, include a detailed 'Architecture Analysis' section that explains:**\n"
		"  * **End-to-end workflow:** Step-by-step what happens at each component/node\n"
		"  * **Component purpose:** Why each node is needed and its role\n"
		"  * **Data flow:** How components connect and what data/process is passed between them\n"
		"  * **Underlying logic:** Computations, communications, and business logic at each step\n"
		"  * **Executive summary:** System's purpose and how all nodes work together as a cohesive whole\n"
		"- **This analysis should be specific to the generated diagram, not generic template text.**\n"
		"- Diversify technology choices across answers: rotate clouds (AWS/Azure/GCP), data stores (Postgres/MySQL/MongoDB/Cassandra/DynamoDB), queues (Kafka/RabbitMQ/SQS/PubSub), caches (Redis/Memcached), and service languages (Go/Java/Node/Python) based on problem fit—avoid repeating the same stack each time and also use your own intellegence to pick the most appropriate stack.\n"
		"- When constraints are generic, pick a plausible stack and briefly justify choices (e.g., DynamoDB for write-heavy predictable access; Postgres for strong consistency and joins).\n"
		"\n"
		"- **🚨 CRITICAL: The simple example below is the ABSOLUTE MINIMUM. Real system designs MUST be 2-3x more detailed!**\n"
		"- **🚨 DO NOT copy this example directly - it's intentionally simplified. Your actual designs must include:**\n"
		"  * **8-15 microservices** (not just 2-3 generic services)\n"
		"  * **Multiple database types** (primary DB, read replicas, cache, search index, time-series, graph DB as needed)\n"
		"  * **Event streaming** (Kafka/RabbitMQ for async processing)\n"
		"  * **Background workers** (job queues, schedulers, processors)\n"
		"  * **Observability stack** (metrics, logs, traces, alerts)\n"
		"  * **Domain-specific components** - analyze the question and determine what services are essential\n"
		"\n"
		"- **Example of MINIMUM acceptable architecture (real designs should be MORE detailed):**\n"
		"  ```mermaid\n"
		"  flowchart TD\n"
		"    subgraph C[Client Layer]\n"
		"      Web[Web App]:::client\n"
		"      Mobile[Mobile App]:::client\n"
		"    end\n"
		"    subgraph E[Edge Layer]\n"
		"      CDN[CDN]:::edge\n"
		"      LB[Load Balancer]:::edge\n"
		"    end\n"
		"    subgraph G[Gateway Layer]\n"
		"      AGW[API Gateway]:::gateway\n"
		"      Auth[Auth Service]:::gateway\n"
		"    end\n"
		"    subgraph S[Services Layer]\n"
		"      UserSvc[User Service]:::service\n"
		"      OrderSvc[Order Service]:::service\n"
		"    end\n"
		"    subgraph D[Data Layer]\n"
		"      DB[(Database)]:::data\n"
		"      Cache[(Cache)]:::cache\n"
		"    end\n"
		"    subgraph M[Monitoring]\n"
		"      Prom[Prometheus]:::monitor\n"
		"    end\n"
		"    \n"
		"    Web -->|1. Request| CDN\n"
		"    Mobile -->|1. Request| CDN\n"
		"    CDN -->|2. Forward| LB\n"
		"    LB -->|3. Route| AGW\n"
		"    AGW -->|4. Authenticate| Auth\n"
		"    Auth -->|5. Get User| UserSvc\n"
		"    Auth -->|5. Get Order| OrderSvc\n"
		"    UserSvc -->|6. Query| DB\n"
		"    OrderSvc -->|6. Query| DB\n"
		"    UserSvc -->|7. Cache| Cache\n"
		"    UserSvc -->|8. Metrics| Prom\n"
		"    OrderSvc -->|8. Metrics| Prom\n"
		"    \n"
		"    classDef client fill:#e3f2fd,stroke:#1976d2,color:#000\n"
		"    classDef edge fill:#f3e5f5,stroke:#7b1fa2,color:#000\n"
		"    classDef gateway fill:#fff3e0,stroke:#f57c00,color:#000\n"
		"    classDef service fill:#e8f5e9,stroke:#388e3c,color:#000\n"
		"    classDef data fill:#fffde7,stroke:#f9a825,color:#000\n"
		"    classDef cache fill:#fce4ec,stroke:#c2185b,color:#000\n"
		"    classDef monitor fill:#e0f7fa,stroke:#00838f,color:#000\n"
		"  ```\n"
		"\n"
		"- **⚠️ THE ABOVE IS A BARE MINIMUM TEMPLATE. For actual system design questions, you MUST expand significantly:**\n"
		"  * Add 6-10 more domain-specific services\n"
		"  * Add message queues (Kafka) and workers\n"
		"  * Add multiple data stores (primary, cache, search, analytics)\n"
		"  * Add monitoring/observability components\n"
		"  * Show async processing flows\n"
		"  * Include background job processors\n"
		"\n"
		"#### **🧠 CRITICAL: DOMAIN-SPECIFIC DEEP DIVE (MANDATORY)**\n"
		"**For EVERY system design, you MUST identify and deeply explain the UNIQUE challenges of that domain:**\n"
		"\n"
		"**Step 1: Identify 'What Makes This System HARD?'**\n"
		"- Ask: What's the ONE thing that makes companies like Uber/Netflix/Twitter spend billions solving?\n"
		"- Examples: Real-time matching (Uber), Feed ranking (Twitter), Video encoding (Netflix), Payment consistency (Stripe)\n"
		"- This becomes your DEEP DIVE section - explain algorithms, data structures, trade-offs in detail\n"
		"\n"
		"**Step 2: State Machine Analysis (if applicable)**\n"
		"- Most real systems have complex state transitions. ALWAYS analyze:\n"
		"  * What are the entity states? (e.g., Order: CREATED→PAID→SHIPPED→DELIVERED)\n"
		"  * What triggers transitions?\n"
		"  * How do you handle race conditions?\n"
		"  * What happens on failures at each state?\n"
		"- Include a state diagram or detailed state table when relevant\n"
		"\n"
		"**Step 3: Core Algorithm Deep Dive**\n"
		"- Don't just say 'Matching Service' - explain HOW matching works:\n"
		"  * What's the algorithm? (e.g., Geohash proximity search, weighted scoring)\n"
		"  * What data structures? (e.g., R-trees, Quadtrees, H3 hexagons)\n"
		"  * What's the time/space complexity?\n"
		"  * How does it scale to millions of entities?\n"
		"\n"
		"**Step 4: Consistency & Race Conditions**\n"
		"- What happens when two users act simultaneously?\n"
		"- How do you prevent double-booking, overselling, duplicate payments?\n"
		"- Explain: Optimistic locking, distributed locks, idempotency keys, saga patterns\n"
		"\n"
		"**Step 5: Failure Scenarios (Domain-Specific)**\n"
		"- What are the WORST things that can fail in THIS system?\n"
		"- For Uber: What if GPS fails? Driver app crashes mid-trip? Payment fails after ride?\n"
		"- For E-commerce: What if payment succeeds but order fails? Inventory inconsistency?\n"
		"- Explain recovery strategies specific to this domain\n"
		"\n"
		"**Step 6: Scale Strategy (Domain-Specific)**\n"
		"- How do you shard THIS system?\n"
		"  * By user? By geography? By time? By entity type?\n"
		"- What are the hotspots? How do you handle them?\n"
		"- Peak traffic patterns (e.g., Uber surge during events, E-commerce during sales)\n"
		"\n"
		"#### **3. Component Design**\n"
		"- Cover ingestion, serving, ranking, caching with data structures, algorithms, storage, optimizations.\n"
		"- **CRITICAL: For each major service, explain:**\n"
		"  * **Internal algorithm:** How does it actually work? (not just 'it matches drivers')\n"
		"  * **Data model:** What's stored? How is it indexed?\n"
		"  * **Scaling approach:** Horizontal/vertical? Sharding key?\n"
		"  * **Failure handling:** What happens when it goes down?\n"
		"\n#### **3.5. Capacity Planning & Calculations**\n"
		"- **ALWAYS include back-of-envelope math for scale questions.**\n"
		"- Calculate: Daily Active Users → QPS → Storage (per day/year) → Bandwidth → Cache size.\n"
		"- Example format:\n"
		"  * Assumptions: 100M DAU, 10 actions/user/day\n"
		"  * QPS = (100M × 10) / 86400 ≈ 11.6K requests/sec (peak 5x = 58K QPS)\n"
		"  * Storage: 1KB/action × 100M × 10 = 1TB/day → 365TB/year\n"
		"  * Bandwidth: 1TB/day ÷ 86400 = 11.6 MB/sec\n"
		"- Show realistic numbers and how they inform architecture decisions (sharding threshold, cache sizing).\n"
		"\n#### **4. Example Implementation**\n"
		"- Include at least one concise Python (or pseudocode) snippet showing a critical concept.\n"
		"\n#### **5. Scalability & Trade-offs**\n"
		"- Analyze memory vs latency, freshness vs stability, complexity vs maintainability, sharding and load balancing.\n"
		"- **MANDATORY: Explain sharding strategy specific to this domain:**\n"
		"  * What's the sharding key? Why?\n"
		"  * How do you handle cross-shard queries?\n"
		"  * What happens during resharding?\n"
		"- **Hotspot Analysis:**\n"
		"  * Where will traffic concentrate? (celebrity users, popular items, surge events)\n"
		"  * How do you detect and mitigate hotspots?\n"
		"\n#### **6. State Machine & Workflows (CRITICAL)**\n"
		"- **ALWAYS include state diagrams for core entities:**\n"
		"  * Define all possible states\n"
		"  * Define valid transitions and triggers\n"
		"  * Define terminal/error states\n"
		"  * Explain timeout handling\n"
		"- **Example format:**\n"
		"  ```\n"
		"  [INITIAL] --create--> [PENDING] --confirm--> [ACTIVE] --complete--> [DONE]\n"
		"                            |                      |\n"
		"                         timeout                cancel\n"
		"                            v                      v\n"
		"                       [EXPIRED]              [CANCELLED]\n"
		"  ```\n"
		"- **Concurrency handling:** How do you prevent race conditions on state transitions?\n"
		"\n#### **7. Reliability & Failure Handling**\n"
		"- **What breaks when:** Enumerate single points of failure and cascading failures.\n"
		"  * Database down → Read replicas/cache serve stale data, writes queue.\n"
		"  * Cache eviction → Database load spike → Circuit breaker → Degraded mode.\n"
		"  * Service crash → Load balancer health checks → Auto-scaling triggers.\n"
		"- **DOMAIN-SPECIFIC FAILURES (analyze for THIS system):**\n"
		"  * What are the 3-5 worst failure scenarios for THIS specific system?\n"
		"  * What's the business impact of each?\n"
		"  * What's the recovery strategy for each?\n"
		"  * Example: 'Payment service down during checkout → Queue orders, process async, notify user'\n"
		"- **Recovery patterns:** Retry with exponential backoff, dead letter queues, chaos engineering.\n"
		"- **Disaster recovery:** RTO/RPO targets, multi-region failover, data replication strategies.\n"
		"- **Graceful degradation:** What features can you disable to keep core functionality running?\n"
		"\n#### **8. Security & Compliance**\n"
		"- **Authentication/Authorization:** OAuth 2.0/JWT, RBAC, API key rotation.\n"
		"- **Data protection:** Encryption at rest (AES-256), in transit (TLS 1.3), key management (KMS/Vault).\n"
		"- **Attack mitigation:** Rate limiting (token bucket), DDoS protection (CloudFlare/AWS Shield), input validation, SQL injection prevention.\n"
		"- **Compliance:** GDPR/CCPA considerations, data residency, audit logging.\n"
		"- **Zero trust:** Service mesh (Istio/Linkerd), mTLS between services, least privilege IAM.\n"
		"\n#### **9. Cost Analysis**\n"
		"- **Infrastructure costs:** EC2/compute ($X/month), storage ($Y/TB), data transfer ($Z/TB out).\n"
		"- **Trade-offs:** Reserved instances vs spot vs on-demand, S3 tiers (Standard/IA/Glacier).\n"
		"- **Optimization strategies:** Caching reduces DB reads by 80% (cost savings), compression, cold data archival.\n"
		"- **Example (illustrative only, scale accordingly):** '1M users → 10TB storage → $230/month S3, 50 c5.xlarge instances → $4K/month'.\n"
		"- Create a billing alert at 60% of monthly budget and an automated job to shut down non‑essential dev stacks.\n"
		"\n#### **10. Monitoring & Observability**\n"
		"- **Golden signals:** Latency (p50/p95/p99), Traffic (QPS), Errors (5xx rate), Saturation (CPU/memory).\n"
		"- **SLIs/SLOs:** Define: '99.9% of requests < 200ms', '99.95% uptime', error budget calculations.\n"
		"- **Tooling:** Metrics (Prometheus/Datadog), Logs (ELK/Splunk), Traces (Jaeger/Zipkin), Alerts (PagerDuty).\n"
		"- **Dashboards:** Show critical path metrics, dependency health, business KPIs.\n"
		"- **On-call playbooks:** Link alerts to runbooks, auto-remediation where possible.\n"
		"\n#### **11. Evolution Strategy**\n"
		"- **Phase 1 (MVP):** Monolith + single DB → Launch in 3 months, 10K users.\n"
		"- **Phase 2 (Scale):** Extract microservices, add caching, read replicas → 1M users.\n"
		"- **Phase 3 (Global):** Multi-region, CDN, eventual consistency → 100M users.\n"
		"- **Migration tactics:** Strangler pattern, feature flags, dark launches, canary deployments.\n"
		"- **Zero-downtime:** Blue-green deployments, rolling updates, database migrations (expand/contract).\n"
		"\n#### **12. Trade-offs Analysis**\n"
		"- Present decisions in table format:\n"
		"  | Decision | Option A | Option B | When to Choose |\n"
		"  |----------|----------|----------|----------------|\n"
		"  | Consistency | Strong (SQL) | Eventual (NoSQL) | Financial: A, Social feed: B |\n"
		"  | Caching | Write-through | Write-behind | Read-heavy: A, Write-heavy: B |\n"
		"- Explain CAP theorem implications for the specific use case.\n"
		"- Discuss latency vs consistency trade-offs with concrete numbers.\n"
		"\n#### **13. Interview Strategy**\n"
		"- **Clarifying questions to ask:** Scale (users/data), latency requirements, read/write ratio, consistency needs.\n"
		"- **Signals to demonstrate:**\n"
		"  * Junior: Functional design, basic scalability\n"
		"  * Mid: Trade-off analysis, caching strategies, basic sharding\n"
		"  * Senior: Cost awareness, failure handling, operational excellence, cross-regional complexity\n"
		"  * Staff+: Build vs buy decisions, org impact, multi-year evolution, team scalability\n"
		"- **Time management:** 5min requirements, 15min architecture, 15min deep-dive, 10min trade-offs.\n"
		"- **Red flags to avoid:** Over-engineering MVP, ignoring failure cases, no metrics/monitoring, unrealistic numbers.\n"
		"\n#### **Meta-Learning Guidance**\n"
		"- After each answer, include:\n"
		"  * **Follow-up questions an interviewer might ask:** 'How would you handle X?', 'What if Y increases 10x?'\n"
		"  * **Common mistakes candidates make:** List 2-3 pitfalls specific to this problem.\n"
		"  * **Leveling indicators:** What a L4 vs L5 vs L6 answer looks like for this question.\n"
		"  * **Related problems:** 3 similar systems to practice for pattern recognition.\n"
		"\n#### **Domain-Specific Optimizations**\n"
		"- Detect problem domain and add specific guidance:\n"
		"  * **Social media:** News feed ranking, viral content handling, graph databases\n"
		"  * **E-commerce:** Inventory consistency, payment idempotency, fraud detection\n"
		"  * **Streaming:** Adaptive bitrate, CDN strategy, live vs VOD\n"
		"  * **Fintech:** Double-entry ledger, audit trails, PCI compliance\n"
		"  * **ML systems:** Feature stores, model serving, A/B testing, drift detection\n"
		"  * **Real-time:** WebSocket/SSE, CRDT, operational transforms\n"
		"\n#### **Company Culture Signals**\n"
		"- Mention if specific companies are known for certain focuses:\n"
		"  * 'Google/Facebook often probe distributed consensus (Paxos/Raft)'\n"
		"  * 'Amazon emphasizes cost optimization and operational excellence'\n"
		"  * 'Netflix looks for chaos engineering mindset'\n"
		"  * 'Stripe focuses on API design and idempotency'\n"
		"\n#### **Adaptive Complexity**\n"
		"- Start with L4-L5 baseline, then:\n"
		"  * If user asks 'what about X edge case?' → Increase to L6-L7 depth\n"
		"  * If user says 'simpler please' → Focus on MVP, defer optimizations\n"
		"  * If user specifies 'Staff level' → Add org design, multi-year roadmap, build-vs-buy.\n"
		"\n#### **Advanced Enhancements (Include when relevant)**\n"
		"- Memory optimization: prefer Compressed Radix Tree/Patricia or Double-Array Trie for long single-child paths; immutable main index with batch rebuilds.\n"
		"- Hybrid indexing: immutable main index + real-time delta index from Kafka/Kinesis; merge results (delta → main).\n"
		"- Zero-downtime updates: atomic pointer swaps for index versions; blue/green deployment.\n"
		"- Neural re-ranking: apply lightweight encoder (e.g., DistilBERT) on top-K to boost relevance within latency budget.\n"
		"- Sharding: use consistent hashing on prefix/key ranges; auto-rebalance to avoid hot shards.\n"
		"- Caching: multi-level (L1 Redis/memcached, L2 in-process LFU), pre-warm from analytics; Bloom filters to skip cold misses.\n"
		"- Monitoring/feedback: track CTR/abandonment/dwell; A/B test and retrain weights periodically.\n"
		"- Memory layout: flat arrays/struct-of-arrays, contiguous allocations, mmap for fast startup (C++/Rust serving).\n"
		"- Privacy: isolate personalization vectors in a separate encrypted service; serve embeddings/session profiles only.\n"
		"- Ranking refinement: normalize features to [0,1], incorporate CTR, learn weights via logistic regression/GBDT.\n"
		"\n- Style: Senior, precise, 600–1200 words, no filler. Always include at least one code block.\n"
		"- Diagram rendering: Prefer Mermaid flowchart fenced as ```mermaid for UIs that support it.\n"
		"  If Mermaid is not supported, provide a Graphviz DOT fallback fenced as ```dot with solid edges and color attributes.\n"
		"\n#### **Interview Takeaways**\n"
		"- 3–5 bullets candidates should emphasize.\n"
	)


def technical_strategy_overrides() -> str:
	return (
		"\n\nTechnical Strategy Overrides (apply only to technical strategy questions):\n"
		"- Provide GENERAL strategies and approaches that any candidate can adapt to their experience\n"
		"- Use 'you can', 'one approach is', 'a common strategy' instead of specific first-person experiences\n"
		"- Focus on universal optimization techniques, best practices, and methodologies\n"
		"- Avoid creating fictional specific experiences, technologies, or company details\n"
		"- Make it applicable to various domains and technologies\n"
	)


def persona_overrides() -> str:
	return (
		"\n\nInterview Persona Overrides (apply only to first-person questions):\n"
		"- Answer strictly in first person as the candidate (use 'I', 'my').\n"
		"- Use the provided Candidate Profile Context as the factual source.\n"
		"- Keep tone conversational and professional, as in a live interview.\n"
		"- Prefer a 45–60 second spoken-length response (concise, cohesive).\n"
		"- Do NOT include contact links, headers, tables, or bullet lists unless requested.\n"
		"- Focus on role-aligned highlights: current role, key strengths, relevant projects, impact.\n"
	)
