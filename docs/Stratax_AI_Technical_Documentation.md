# Stratax AI
## Backend Technical Architecture & System Design

---

**Version:** 2.0  
**Date:** February 2026  
**Organization:** Stratax AI Development Team  
**Classification:** Technical Architecture Document

---

<div style="page-break-after: always;"></div>

# Executive Summary

## Overview

Stratax AI is an interview simulation and evaluation engine designed to model real interviewer behavior and generate structured, explainable assessment reports. Unlike traditional interview prep platforms focused on question banks or competitive coding, Stratax evaluates reasoning quality, communication clarity, and decision-making at the session level.

## Problem Statement

Structured interview simulation and evaluation faces several critical challenges:
- **Lack of personalized feedback** on communication delivery and technical accuracy
- **Limited access** to company-specific and domain-targeted interview questions
- **Insufficient real-time coaching** during practice sessions
- **No systematic evaluation** of coding solutions with actionable improvement suggestions

## Solution Architecture

Stratax AI addresses these challenges through a multi-layered backend architecture:

1. **Intelligent Q&A Engine** — Context-aware conversational assistant powered by Groq and Gemini LLMs
2. **Practice Mode** — Real-time audio-based interview simulator with speech analytics and micro-feedback
3. **Interview Intelligence** — Vector search and LLM-driven question generation with optional web augmentation
4. **Code Evaluation** — Static analysis combined with LLM critique for comprehensive code review
5. **Architecture Generation** — Automatic system diagram creation from code analysis

## Key Capabilities

- **Multi-provider LLM abstraction** supporting Groq and Google Gemini with hot-swappable configuration
- **Local speech processing** using faster-whisper (STT) and pyttsx3/gTTS (TTS) for offline practice
- **Hybrid retrieval system** combining BM25 lexical search with semantic vector search via Qdrant
- **Agent-based architecture** with specialized components for interviewing, evaluation, and speech analytics
- **Company-specific interview intelligence** with targeted question generation
- **Real-time feedback loops** with micro-coaching on delivery metrics (pace, fillers, confidence)
- **Telemetry-driven personalization** using a structured event stream (privacy-safe stable hashes) and deterministic learning loops for recommended practice focus areas

## Technology Foundation

**Runtime:** Python 3.11+ + FastAPI + Starlette  
**AI/ML:** Groq SDK, Google Generative AI, sentence-transformers, faster-whisper  
**Storage:** SQL DB via `DATABASE_URL` (SQLite or Postgres) for structured records (users/usage/rate limits/telemetry), Qdrant vector DB for semantic retrieval, plus file-based JSON/JSONL persistence for sessions/audit logs. SQLite is supported for local development only; production deployments are expected to use PostgreSQL (e.g., Neon).  
**Deployment:** Docker containerization with docker-compose orchestration (defaults to Postgres + Qdrant + Redis)  
**Audio:** librosa, soundfile, pyttsx3, gTTS

## Target Users

- **Engineering Teams** integrating AI interview capabilities into products
- **Platform Developers** building interview simulation and evaluation services
- **Technical Architects** evaluating AI-driven coaching systems
- **DevOps Engineers** deploying and scaling the service

## Who This System Is For (And Who It Is Not)

### Target users

- Hiring teams conducting mock or structured interviews
- Internal L&D teams evaluating interview readiness
- Bootcamps and training institutes running interview simulations
- Recruitment agencies performing candidate screening

### Individual use (students & tech professionals)

Individual candidates (students and working professionals) use Stratax to:

- experience realistic interview simulations
- practice explaining reasoning, not just coding
- receive structured feedback similar to real interviews
- identify gaps in communication, problem-solving, and approach

This individual usage mirrors how the system is used by hiring teams and training programs, making it suitable both for self-directed practice and formal evaluation.

**The same evaluation and reporting pipeline is used for both individual practice and institutional assessment.**

### Non-goals

- Stratax is not a competitive programming platform (e.g., LeetCode)
- It does not focus on large-scale coding challenge libraries
- It is not designed for mass B2C, consumer-scale self-serve practice

### Positioning

Stratax is an **Interview Simulation and Evaluation Engine**, focused on:

All subsystems are designed to support a single goal: producing realistic interview simulations and reliable evaluation signals.

While Stratax is used by individual candidates, its evaluation and feedback system improves through aggregated, anonymized practice data rather than static question banks.

- interviewer-style questioning
- reasoning and explanation quality
- session-level intelligence
- structured evaluation reports
- fairness, consistency, and auditability

## Learning From Individual Interview Practice

Stratax is designed for students and tech professionals preparing for interviews.
Each practice session generates anonymized behavioral and performance signals such as:

- communication pace and hesitation
- reasoning depth and follow-up quality
- confidence self-assessment after interviews

These signals allow Stratax to identify patterns of improvement, common mistakes, and confidence progression across repeated practice sessions.

Over time, this enables more accurate feedback and more realistic interview simulations, even for individual users.

<div style="page-break-after: always;"></div>

# System Overview

## Backend Responsibilities

The Stratax AI backend serves as the central orchestration layer for all interview intelligence operations. Core responsibilities include:

### Primary Functions

1. **Session Management** — Per-user conversational context with persistent history
2. **LLM Orchestration** — Provider-agnostic request routing with fallback handling
3. **Vector Search** — Semantic question retrieval using embedding-based similarity
4. **Audio Processing** — Transcription, synthesis, and delivery metric extraction
5. **Evaluation Pipeline** — Multi-stage assessment of answers and code submissions
6. **Diagram Generation** — Architecture visualization from code analysis

## Recent Updates (Past Week)

### Resume Parsing & Document Analysis (NEW)

- Added **resume parser** (`app/services/core/resume_parser.py`) with LLM-based structured extraction from PDF/DOCX/TXT uploads.
- Parsed resumes feed claim-based probing: interview questions target stated project claims, skills, and achievements.
- Added **document analyzer** (`app/services/core/document_analyzer.py`) with ATS scoring, skills-gap analysis, career trajectory insights, and industry benchmarking at four analysis depths (Quick → Expert).
- Automatic document type detection via configurable keyword heuristics.

### Mirror Ontology, Progress Tracking & Dynamic Budget (NEW)

- **Mirror Ontology Generator** (`app/services/chat/mirror_ontology.py`) — LLM-generated ontologies (primitives, senior signals, red flags) with two-layer cache (in-memory + Redis).
- **Mirror Progress Comparison** (`app/services/chat/mirror_compare.py`) — cross-attempt diff of gaps closed, confidence delta, and red flags resolved.
- **Dynamic Budget Engine** (`app/services/chat/dynamic_budget.py`) — adaptive per-request token budgeting replacing hardcoded limits, using intent × depth × user-tier multipliers.

### Practice Mode Enhancements (NEW)

- **Deterministic Scoring** (`app/services/practice/practice_scoring.py`) — rubric-based practice scoring without LLM.
- **Adaptive Pressure** (`app/services/practice/adaptive_pressure.py`) — difficulty auto-adjustment based on running performance.
- **Practice Progress & Learning** (`app/services/practice/practice_progress.py`, `practice_learning.py`) — cross-session progress tracking and learning-loop insights.
- **Round Config Service** (`app/services/practice/round_config_service.py`) — centralized round selection logic.
- **LangGraph Orchestration** (`app/services/practice/practice_mode_graph.py`) — optional graph-based practice flow (behind `ENABLE_LANGGRAPH_PRACTICE` flag).

### Mock Interview Analytics (NEW)

- Added `app/services/interview/mock_interview_analytics.py` for within-session trajectory computation (per-criterion deltas, direction detection).

### Infrastructure: Redis, Sentry, Gemini Adapter (NEW)

- **Redis client** (`app/services/core/redis_client.py`) — optional Redis for distributed caching and rate limiting.
- **Sentry integration** (`app/services/core/sentry.py`) — automatic error tracking with `SENTRY_DSN`.
- **Gemini adapter** (`app/services/llm/gemini_adapter.py`) — direct Gemini model adapter with content-safety mapping.
- **Usage tracking** (`app/utils/usage_tracking.py`) — per-session token usage tracking.

### Response formatting & structural integrity (NEW)

- Built a comprehensive **response post-processing pipeline** (`app/services/llm/response_postprocess.py`, ~1,750 lines) that sanitizes all LLM output before it reaches the UI.
- Pipeline stages: bullet normalization, code-fence-aware markdown repair, Mermaid block sanitization, colon-label list conversion, emphasis balancing, prompt-leak stripping, and more.
- Added **per-chunk stream sanitization** so streamed responses get real-time cleanup (unicode bullets, smart quotes, dashes) before each SSE chunk is sent to the client.
- Added a **structural integrity validator** as the final assertion layer: verifies balanced code fences, balanced emphasis markers, Mermaid block hygiene (valid header + non-empty), and response size limits (32 KB with clean sentence-boundary truncation). Auto-repairs all violations.
- Strengthened `RESPONSE_TEMPLATE` and `CODE_QUALITY` system prompt policies with explicit code-block, bold, table, and anti-truncation rules.

### Modular prompt architecture (NEW)

- Introduced a **PolicyModule system** (`app/prompts/policies.py`) with named, composable prompt modules: `RESPONSE_CONTRACT`, `RESPONSE_TEMPLATE`, `CODE_QUALITY`, `SYSTEM_DESIGN`, `COPILOT_SYSTEM`, `MIRROR_MODE`, and others.
- `app/prompts/builder.py` provides `build_default_system_prompt()` which composes policies based on `PromptFlags` (question type, domain, routing intent).
- Mirror mode uses a completely separate policy set via `app/prompts/mirror_policies.py`.
- Response planning via `app/prompts/response_plan.py` defines a `ResponsePlan` dataclass for structured LLM output.

### Cross-session semantic deduplication (NEW)

- `AdaptiveInterviewerAgent` now performs **semantic deduplication** of generated questions using `sentence-transformers` (`all-MiniLM-L6-v2`).
- Cross-session memory prevents repeating questions across practice sessions.
- Intra-batch cosine-similarity dedup ensures variety within a single generation batch.
- Rejection metrics logged for observability.

### Telemetry & learning loops

- Added a structured telemetry spine using a database-backed `event_records` stream (SQLAlchemy `EventRecord`).
- Events use privacy-safe stable identifiers (SHA256 or HMAC-SHA256 when configured) and avoid raw text storage by default.
- Introduced deterministic “learning loops” that aggregate recent practice events into explainable insights.
- Exposed `GET /api/practice/insights` to return summary stats and `recommended_focus` for personalization.

### Interview Intelligence: key policy + history correctness

- Enforced `REQUIRE_USER_API_KEY` behavior: when enabled, Interview Intelligence requests without a client-supplied key return **401** (no silent fallback).
- Prevented duplicate history entries by disabling backend auto-save for standard Interview Intelligence search; the UI should save exactly once through the History API.

### Code Studio execution + Visualize (backend sandbox)

- Added a dedicated LeetCode-style execution endpoint: `POST /api/code/execute`.
- Execution is server-side via sandbox providers (Judge0 via RapidAPI with optional fallback providers).
- Supports `stdin`, optional `test_cases`, and tracing.
- Tracing can optionally attach per-line explanations (`explain_trace`) to support a “Visualize” UI.

### Production secrets + encrypted provider keys

- Production now requires persistent `JWT_SECRET_KEY` and `COOKIE_SECRET` (to avoid token/cookie invalidation on restart).
- Added encryption for user-stored provider keys using `STRATAX_SECRETS_ENCRYPTION_KEY` (Fernet), stored in DB with an `enc:` prefix.
- Backward compatible: plaintext rows can still be read, but production refuses to store new provider keys unless encryption is configured.

### CI (GitHub Actions)

- `.github/workflows/ci.yml` runs tests on push/PR (Python 3.11/3.12 matrix), produces coverage, and runs lint/format/security checks.
- CI sets safe flags (`FAST_STARTUP=true`, `DISABLE_INTERVIEW_INTELLIGENCE=true`, `ENABLE_CODE_EXECUTION=false`) and uses fixed test secrets.

### Service Architecture

The system is designed as a **monolithic FastAPI service** with optional modular components:

- **Always Available:** Q&A engine, session management, health endpoints
- **New:** Auth subsystem (JWT + optional Google OAuth) backed by the configured SQL DB (`DATABASE_URL`; SQLite or Postgres)
- **New:** Tier-based rate limiting middleware (plus demo-mode quotas + debug headers)
- **Optional Components:**
  - Interview Intelligence (requires Qdrant and sentence-transformers)
  - Practice Mode (requires audio libraries and TTS/STT models)
  - Code Execution Sandbox (requires Judge0 or Piston API)

### Deployment Model

**Single Service Container** running on port 7860 with:
- Worker count controlled by `UVICORN_WORKERS` (defaults to 1 in Dockerfile; docker-compose uses 2)
- Lifespan-managed initialization of shared components
- File-system persistence for sessions and vector indices
- Optional Kroki sidecar for Mermaid rendering

### Subsystem Organization

```
FastAPI Application
├── Core Q&A Engine
│   ├── Session Manager (app/services/core/session_manager.py)
│   ├── LLM Service (app/services/chat/llm_service.py)
│   ├── History Manager (app/services/core/history_manager.py)
│   └── Response Post-Processing (app/services/llm/response_postprocess.py)
│
├── Prompt Architecture
│   ├── PolicyModule System (app/prompts/policies.py)
│   ├── Prompt Builder (app/prompts/builder.py)
│   ├── Mirror Policies (app/prompts/mirror_policies.py)
│   └── Response Plan (app/prompts/response_plan.py)
│
├── LLM Intelligence Layer
│   ├── Identity Guard (app/services/llm/identity.py)
│   ├── Intent Overrides (app/services/llm/intent_overrides.py)
│   ├── Groq Model Routing (app/services/llm/groq_models.py)
│   └── Structural Integrity Validator (in response_postprocess.py)
│
├── Practice Mode (Optional)
│   ├── Interview Orchestrator (app/services/practice/practice_mode_service.py)
│   ├── Speech Analytics Agent (app/services/practice/speech_analytics_agent.py)
│   ├── Adaptive Interviewer Agent (app/services/practice/adaptive_interviewer_agent.py)
│   ├── Conversational Agent (app/services/practice/conversational_agent.py)
│   ├── Evaluation Agent (app/services/practice/evaluation_agent.py)
│   ├── Local STT Service (app/services/practice/local_stt_service.py)
│   ├── Local TTS Service (app/services/practice/local_tts_service.py)
│   ├── Semantic Deduplication (in adaptive_interviewer_agent.py)
│   ├── Deterministic Scoring (app/services/practice/practice_scoring.py)
│   ├── Adaptive Pressure (app/services/practice/adaptive_pressure.py)
│   ├── Practice Progress (app/services/practice/practice_progress.py)
│   ├── Practice Learning (app/services/practice/practice_learning.py)
│   ├── Round Config (app/services/practice/round_config_service.py)
│   ├── LangGraph Orchestration (app/services/practice/practice_mode_graph.py) [optional]
│   └── Learning Loops (app/services/practice/learning_loops.py)
│
├── Interview Intelligence (Optional)
│   ├── Vector Store — Qdrant + embeddings (app/services/interview/interview_intelligence_service.py)
│   ├── Question Generator (LLM)
│   ├── Hybrid Search Engine — BM25 + semantic (app/services/chat/ai_native_enhancements.py)
│   ├── Cohere Reranker (optional)
│   ├── Query Expansion (LLM-based)
│   └── Code Execution Sandbox (optional)
│
├── Mock Interview
│   ├── Session Persistence (app/services/interview/mock_interview_service.py)
│   ├── Session Analytics (app/services/interview/mock_interview_analytics.py)
│   ├── Progressive Hints
│   └── LLM Evaluation
│
├── Resume & Document Intelligence
│   ├── Resume Parser (app/services/core/resume_parser.py)
│   └── Document Analyzer (app/services/core/document_analyzer.py)
│
├── Mirror Mode
│   ├── Ontology Generator (app/services/chat/mirror_ontology.py)
│   └── Progress Comparison (app/services/chat/mirror_compare.py)
│
├── LLM Cost & Budget
│   └── Dynamic Budget Engine (app/services/chat/dynamic_budget.py)
│
├── Infrastructure
│   ├── Redis Client (app/services/core/redis_client.py)
│   ├── Sentry Integration (app/utils/sentry.py)
│   ├── Usage Tracking (app/utils/usage_tracking.py)
│   ├── Email Sender (app/utils/email_sender.py)
│   └── Gemini Adapter — lazy import proxy (app/services/chat/gemini_adapter.py)
│
├── Code Evaluation
│   ├── Static Analysis (app/services/core/code_evaluation_service.py)
│   ├── LLM Critique
│   └── In-memory Cache
│
└── Architecture/Diagram Generator
    ├── Mermaid Sanitization (app/utils/mermaid_sanitizer.py)
    ├── Complexity Detection (app/services/architecture/architecture_generator.py)
    └── Optional Kroki Rendering
```

<div style="page-break-after: always;"></div>

# Architecture Overview

## High-Level Component Interaction

```mermaid
graph TB
    Client["<b>Client Layer</b><br/>Web UI, Mobile App, API Consumers"]
    
    subgraph Gateway["API Gateway Layer - FastAPI + Middleware"]
        MW["CORS | Auth | User ID Extraction | Rate Limiting | Audit"]
    end
    
    subgraph Core["Core Service Layer"]
        QA["<b>Q&A Core</b><br/>• SessionManager<br/>• LLM Service<br/>• History"]
        PM["<b>Practice Mode</b><br/>• 5 Agents<br/>• STT/TTS"]
        IE["<b>Intelligence Engine</b><br/>• Vector DB<br/>• LLM Generation"]
    end
    
    subgraph Infrastructure["Infrastructure Layer"]
        LLM["<b>LLM Layer</b><br/>• Groq<br/>• Gemini"]
        Audio["<b>Audio Layer</b><br/>• faster-whisper<br/>• pyttsx3/gTTS"]
        Vector["<b>Vector Layer</b><br/>• Qdrant<br/>• sentence-transformers"]
    end
    
    Persist["<b>Persistence Layer</b><br/>• SQL DB (SQLite dev / Postgres prod): users/usage/rate limits/telemetry<br/>• Sessions: JSON per user<br/>• Vector DB: Qdrant local store<br/>• Audio: WAV files (session only)<br/>• History/Audit: JSONL logs"]
    
    Client -->|HTTP/WebSocket| Gateway
    Gateway --> QA
    Gateway --> PM
    Gateway --> IE
    
    QA --> LLM
    PM --> LLM
    PM --> Audio
    IE --> Vector
    IE --> LLM
    
    LLM --> Persist
    Audio --> Persist
    Vector --> Persist
```

## Primary Use Case: Structured Interview Simulation for Hiring

1. An interview session is created.
2. The candidate completes a simulated interview (questions + follow-ups).
3. The system captures:
    - answers
    - reasoning/explanations
    - timing (e.g., pace/pauses)
    - proctoring events (where enabled)
4. The system produces:
    - evaluation scores
    - strengths & gaps
    - red flags
    - an interview readiness signal

This mirrors how real interviews are evaluated, not how coding platforms rank users.

## Request Flow Example: Practice Mode Audio Submission

```mermaid
graph TD
    A["User uploads audio"] --> B["Router authenticates & validates"]
    B --> C["PracticeModeService orchestrator"]
    C --> D["LocalSTTService<br/>transcribes audio<br/>(faster-whisper)"]
    D --> E["SpeechAnalyticsAgent<br/>extracts metrics<br/>(WPM, fillers, pauses)"]
    E --> F["InterviewerAgent<br/>generates micro-feedback<br/>(optional: Gemini)"]
    F --> G["AdaptiveInterviewerAgent<br/>evaluates comprehensively<br/>(shared llm_service)"]
    G --> H["Response assembled<br/>with feedback + next question"]
```

## Data Flow: Interview Intelligence Search

```mermaid
graph TD
    Q["User query:<br/>'Python concurrency interview questions'"]
    Q --> R["Router → InterviewIntelligenceService"]
    R --> QE["QueryExpansion<br/>(LLM augments query)<br/><i>Optional</i>"]
    QE --> HS["HybridSearchEngine"]
    HS --> BM25["BM25 lexical retrieval"]
    HS --> SEM["Semantic search<br/>(Qdrant + embeddings)"]
    BM25 --> RR["CohereReranker<br/>reorders candidates<br/><i>Optional</i>"]
    SEM --> RR
    RR --> LG["LLM generates fresh questions<br/><i>Optional</i>"]
    LG --> RES["Results returned with<br/>metadata + code solutions"]
```

<div style="page-break-after: always;"></div>

# Technology Stack

## Runtime & Framework

### Core Platform
- **Python 3.11+** — Modern async/await support; CI runs on Python 3.11 and 3.12
- **FastAPI** — High-performance async web framework with automatic OpenAPI documentation
- **Starlette** — ASGI toolkit for routing, middleware, and WebSocket support
- **Uvicorn** — Lightning-fast ASGI server for production deployment
- **Pydantic** — Data validation and settings management with type safety

### Configuration Management
- **pydantic-settings** — Environment-based configuration with `.env` support
- **python-dotenv** — Secure secrets management via environment variables

## AI & Machine Learning

### Large Language Models
- **Groq SDK** — Ultra-fast inference with Llama and Mixtral models
- **Google Generative AI (Gemini)** — Multimodal capabilities, large context windows

*The system provides a unified abstraction layer allowing hot-swappable provider selection via configuration.*

### Embeddings & Vector Search
- **sentence-transformers** — State-of-the-art semantic embeddings (all-MiniLM-L6-v2)
- **torch (CPU build)** — PyTorch runtime optimized for CPU inference
- **Qdrant** — High-performance vector database with HNSW indexing

### Retrieval Augmentation
- **LangChain Community** — BM25 retrieval, vectorstore abstractions, text splitters
- **rank-bm25** — Lexical search for hybrid retrieval
- **Cohere API** — Neural reranking for search result optimization

### Speech Processing
- **faster-whisper** — OpenAI Whisper optimized for CPU with CTranslate2
- **pyttsx3** — Offline text-to-speech (Windows SAPI, macOS NSSpeechSynthesizer, Linux espeak)
- **gTTS** — Google Text-to-Speech fallback for cloud-based synthesis
- **librosa** — Audio feature extraction for speech analytics
- **soundfile** — Audio I/O operations

### Agent Architecture
The Practice Mode subsystem implements a multi-agent pattern:
- **InterviewerAgent** — Question sequencing and micro-feedback
- **SpeechAnalyticsAgent** — Delivery metrics via DSP (not LLM-based)
- **AdaptiveInterviewerAgent** — LLM-driven answer evaluation
- **ConversationalAgent** — Profile inference from natural language
- **EvaluationAgent** — End-of-interview coaching reports

## Storage & Persistence

### Primary Storage
- **SQL DB (`DATABASE_URL`)** — Users/auth/telemetry/usage/rate limits (SQLite for local/dev; Postgres for production)
- **File System (optional)** — JSON/JSONL for sessions and history artifacts (dev-friendly)
- **Qdrant** — Vector database for interview questions (server via `QDRANT_URL` or local-path fallback for single-process dev)

### Persistence Strategy
- User sessions: stored in the SQL DB by default when using Postgres, or in files at `data/sessions/{user_id}/{session_id}.json` when `STRATAX_SESSION_STORE=file` (common in SQLite/dev)
- Vector indices: `data/interview_intelligence_v2/vector_db/`
- Audio recordings: `data/practice_audio/` (WAV format). Audio files are used for session-level processing and user playback only and are not used for learning or analytics.
- Model cache: `data/models/` (sentence-transformers downloads)

*Note: The system uses a hybrid approach: structured records live in the SQL DB, while some artifacts remain file-backed for dev simplicity. Remaining file-backed pieces can be migrated to Postgres/object storage for scale.*

## External Integrations (Optional)

- **Serper API** — Web search for real-time interview question sourcing
- **Judge0/Piston** — Code execution sandboxes for safe code evaluation
- **GitHub API** — Repository search for technical question curation
- **Kroki** — Mermaid diagram rendering service

## Containerization

- **Docker** — Single-service container with Python 3.11 slim base (`python:3.11-slim`)
- **docker-compose** — Multi-container orchestration (app + Postgres + Qdrant + Redis + optional Kroki)

<div style="page-break-after: always;"></div>

# Core Capabilities & Modules

## 1. Q&A Engine

### Purpose
Conversational AI assistant for interview simulation with persistent session context and multi-turn conversation support.

### Key Responsibilities
- Session lifecycle management (create, list, delete, title updates)
- LLM request orchestration with provider selection
- Conversation history persistence per user
- Profile upload and context enrichment (resume/job description analysis)

### Architecture Pattern
```
User Request → Router → SessionManager → LLM Service → Response
                  ↓
            Per-user JSON persistence
```

### Notable Features
- **API key flexibility:** Accepts user-supplied keys or falls back to server configuration
- **Streaming support:** Optional response streaming for real-time UX
- **Identity guard:** System prompt enforces attribution to prevent AI impersonation

---

## 2. Practice Mode

### Purpose
Real-time audio-based interview simulator with comprehensive speech analytics, micro-feedback, and end-to-end evaluation.

When `ENABLE_PRACTICE_LEARNING` is enabled, Practice Mode optionally captures anonymized behavioral metrics (e.g., speaking pace, hesitation, follow-up depth) and a post-session confidence score provided by the user. These signals are aggregated across sessions to improve feedback quality over time. No raw audio or transcripts are stored.

### Key Responsibilities
- **Round-based interviews:** Pre-configured rounds (HR, Technical, System Design) with dynamic question generation
- **Quick-start onboarding:** Conversational AI infers user profile from natural language
- **Audio processing:** Local STT transcription with Voice Activity Detection (VAD)
- **Speech analytics:** Delivery metrics (WPM, fillers, pauses, pitch variance, confidence scoring)
- **Micro-feedback:** Immediate coaching tips (< 15 words each) on delivery
- **Comprehensive evaluation:** Per-answer technical accuracy scoring with actionable suggestions
- **Final report:** End-of-interview coaching with strengths, improvements, and action plan

### Multi-Agent Architecture
```
PracticeModeService (Orchestrator)
    ├─ LocalSTTService → Transcription
    ├─ SpeechAnalyticsAgent → Delivery metrics
    ├─ InterviewerAgent → Question bank + micro-feedback
    ├─ AdaptiveInterviewerAgent → Answer evaluation (LLM)
    ├─ ConversationalAgent → Profile inference (LLM)
    └─ EvaluationAgent → Final coaching report (LLM)
```

### Workflow
1. **Start:** User initiates via quick-start or round selection
2. **Question Presentation:** Question delivered with optional TTS audio
3. **Answer Submission:** User uploads audio answer
4. **Processing:** STT → analytics → evaluation → micro-feedback
5. **Acknowledgment Gate:** User acknowledges feedback before next question
6. **Completion:** Final evaluation report generated with comprehensive coaching

### Learning Insights

When practice learning is enabled and a minimum cohort size is met (currently 3+ sessions in the same peer bucket), evaluation reports may include a learning insight derived from aggregated, anonymized practice data. Insights are omitted for small cohorts to avoid misleading feedback.

### Differentiators
- **Local processing:** No cloud STT dependency (faster-whisper runs offline)
- **Dynamic filler detection:** Not hardcoded lists; detects repetitions, fragments, false starts
- **Company-specific interview intelligence:** Generates questions tailored to Google, Meta, Amazon, etc.

## What Makes This Different

- Session-level evaluation instead of question-level scoring
- Interviewer-style follow-up logic
- Feedback is informed by real practice behavior, not just LLM-generated answers
- Confidence and communication signals are tracked across sessions for the same user
- LLM-provider-aware evaluation (Groq/Gemini routing)
- Proctoring signal capture (focus, tab switches, presence)
- Explainable evaluation output (not just pass/fail)

### Core Differentiation

- Interviewer-style follow-up logic (not static Q&A)
- Session-level evaluation (not per-question gamification)
- Explainable scoring (strengths, gaps, red flags)
- Proctoring signals integrated into evaluation context
- Same engine used for self-practice and formal assessment

---

## 3. Interview Intelligence

### Purpose
Semantic search and intelligent question generation system powered by vector embeddings and LLM augmentation.

### Key Responsibilities
- **Vector search:** Qdrant-based semantic retrieval with embedding similarity
- **Question generation:** LLM creates questions when database lacks sufficient results
- **Code solution backfill:** Automatically generates code solutions for technical questions
- **Source transparency:** Tracks question provenance (curated, generated, community)

### AI-Native Enhancement Pipeline
When feature flags enabled:
```
Query Input
    ↓
[QueryExpansion] ← LLM augments query
    ↓
[HybridSearchEngine] ← BM25 + Semantic fusion
    ↓
[CohereReranker] ← Neural reranking
    ↓
[CodeExecutionSandbox] ← Validate solutions
    ↓
Results + Metadata
```

### Hybrid Retrieval Strategy
- **Lexical (BM25):** Exact keyword matching for specific terms (e.g., "asyncio", "React hooks")
- **Semantic (Vector):** Conceptual similarity for natural language queries
- **Fusion:** Reciprocal rank fusion combines scores for optimal ranking

### Optional Enhancements
- **Web augmentation:** Serper API fetches real-time questions from web sources
- **Code execution:** Judge0/Piston validates code solutions in sandboxed environments
- **Streaming search:** WebSocket-based progressive result delivery

---

## 4. Mock Interview

### Purpose
Multi-question interview session simulator with progressive difficulty, hints, and LLM-based evaluation.

### Key Responsibilities
- Question sequence management (typically 5 questions per session)
- Progressive hint system (3 levels: gentle nudge → concept clarification → near-solution)
- Per-question timing and evaluation tracking
- Final summary with scores and improvement recommendations

### Persistence Model
- Active sessions: In-memory with periodic JSON persistence
- History: Long-term storage per user in consolidated JSON file

### Evaluation Flow
```
User submits answer
    ↓
LLM evaluates with temporarily increased token limit
    ↓
Score (0-100) + detailed feedback
    ↓
Persisted to session
```

---

## 5. Code Evaluation

### Purpose
Hybrid static + LLM analysis for code submissions with caching to avoid redundant evaluations.

### Key Responsibilities
- **Static analysis:** Code length, complexity heuristics, language detection
- **LLM critique:** Detailed feedback on correctness, style, optimization, edge cases
- **Context-aware classification:** Determines if code evaluation is appropriate for given conversation
- **Result caching:** In-memory cache keyed by session + context + code hash

### Architecture
```
Code Submission
    ↓
[Static Signals] → Metrics (lines, complexity)
    ↓
[LLM Service] → Detailed critique
    ↓
[Cache] → Store result
    ↓
Response with evaluation
```

---

## 6. Architecture & Diagram Generation

### Purpose
Automatic system architecture diagram generation from codebase analysis with Mermaid syntax output.

### Key Responsibilities
- **Complexity detection:** Analyzes code for architecture-worthy patterns (microservices, layering, event-driven)
- **Multi-view generation:** Creates Component, Deployment, Sequence, and Data Flow diagrams
- **Mermaid sanitization:** Strips CSS/HTML artifacts and repairs syntax errors
- **Optional rendering:** Kroki service integration for SVG/PNG export

### View Types
- **Component View:** High-level system components and relationships
- **Deployment View:** Infrastructure and hosting topology
- **Sequence Diagram:** Request/response flows for key operations
- **Data Flow:** Information movement through system layers

---

## 7. Resume Parsing & Claim-Based Interview Probing (NEW)

### Purpose
Extract structured data from uploaded resumes for privacy-safe, claim-based interview probing. Enables targeted questioning on specific project claims, achievements, and stated experience.

### Key Responsibilities
- **Multi-format support:** PDF, DOCX, plain text extraction
- **LLM-based structured parsing:** Extracts skills, projects (with tech + claims), experience summary, role titles, education, achievements, years of experience, and primary domain
- **Regex fallback:** Deterministic extraction when LLM is unavailable
- **Privacy-safe design:** Raw file is deleted after extraction; only structured `ResumeParseResult` persists in-memory for the session duration

### Architecture
```
Resume Upload (PDF/DOCX/TXT)
    ↓
[Text Extraction] → Raw text (max 6,000 chars for LLM)
    ↓
[LLM Structured Parsing] → JSON with skills, projects, claims
    ↓ (fallback)
[Regex Extraction] → Basic skills/education/experience
    ↓
ResumeParseResult → ResumeContext (Pydantic schema)
    ↓
Injected into Practice/Mock Interview prompts
    ↓
Claim-based follow-up questions
```

### Schema Extensions
- `ResumeProject` — structured project with name, tech stack, and claims
- `ResumeContext` — holds parsed resume data
- `UserProfile.resume_context` — optional field for enriched sessions
- `RoundSelectionRequest.resume_context` / `QuickStartRequest.resume_context` — enable resume-aware starts

### Key File
- `app/services/core/resume_parser.py` (~350 lines)

---

## 8. Document Analysis (NEW)

### Purpose
Multi-dimensional document analysis engine surpassing basic keyword matching — supports resumes, job descriptions, cover letters, and LinkedIn profiles.

### Key Responsibilities
- **ATS compatibility scoring** (0-100) with specific recommendations
- **Industry benchmarking** and competitive positioning analysis
- **Skills gap identification** with learning paths
- **Career trajectory analysis** from work history patterns
- **Formatting quality and readability scoring**

### Analysis Depths
| Depth | Time | Description |
|-------|------|-------------|
| Quick | ~30s | Overview only |
| Standard | 1-2 min | Comprehensive analysis |
| Deep | 3-5 min | Exhaustive with recommendations |
| Expert | 5+ min | Industry expert-level assessment |

### Document Type Detection
Uses configurable keyword heuristics (`document_jd_keywords`, `document_resume_keywords`, `document_cover_letter_keywords` in `app/config.py`) for automatic document classification.

### Key File
- `app/services/core/document_analyzer.py` (~658 lines)

---

## 9. Mirror Ontology & Progress Tracking (NEW)

### Purpose
Powers the Mirror Mode comparison feature with LLM-driven ontology generation and cross-attempt progress tracking.

### Mirror Ontology Generator
For a given interview question, generates:
- **Topic** — identified subject area
- **Primitives** — core concepts expected in a good answer
- **Senior signals** — indicators of senior-level understanding
- **Red flags** — warning signs of poor understanding
- **Likely follow-ups** — expected follow-up questions

Uses a two-layer cache: in-memory TTL/LRU + optional Redis (via `app/services/core/redis_client.py`).

### Mirror Progress Comparison
`compute_mirror_progress()` computes the diff between successive mirror reports:
- Gaps closed / new gaps
- Confidence delta
- New strengths
- Red flags resolved / new red flags

### Key Files
- `app/services/chat/mirror_ontology.py` (~193 lines)
- `app/services/chat/mirror_compare.py` (~137 lines)

---

## 10. Dynamic Budget Engine (NEW)

### Purpose
Adaptive token budget computation that replaces hardcoded token buckets. Controls LLM cost and response quality per-request.

### Formula
```
tokens = token_per_budget_unit × intent_unit × depth_multiplier × length_scale × user_tier_multiplier
```

### Budget Multipliers
| Dimension | Values |
|-----------|--------|
| Intent | general (1.0), system_design (2.0), coding (1.5), mirror (1.2), greeting (0.5) |
| Depth | quick (0.6), standard (1.0), deep (1.6) |
| User Tier | free (0.8), standard (1.0), pro (1.25), internal (1.5) |

Result is clamped to configured ceilings. Singleton: `dynamic_budget_engine`.

### Key File
- `app/services/chat/dynamic_budget.py` (~80 lines)

---

## 11. Mock Interview Analytics (NEW)

### Purpose
Deterministic within-session trajectory computation for Mock Interview sessions.

### Key Capabilities
- **Per-criterion score tracking:** correctness, completeness, clarity, confidence, technical_depth
- **Trajectory computation:** start/end/delta per criterion
- **Direction detection:** Improving / Stable / Declining based on overall score delta
- **Best improvement criterion:** identifies which skill improved most

### Key File
- `app/services/interview/mock_interview_analytics.py` (~151 lines)

<div style="page-break-after: always;"></div>

# AI & Intelligence Layer

## Response Post-Processing Pipeline (NEW)

### Purpose
Enterprise-grade formatting and validation layer that ensures every LLM response reaches the UI as clean, well-structured markdown — regardless of provider quirks, mid-stream truncation, or hallucinated formatting.

### Architecture
```
Raw LLM Output
    ↓
[Bullet Normalization] → Unicode •/·/‣ → hyphen bullets
    ↓
[Colon-Label Lists] → "Label: value" → "- **Label:** value"
    ↓
[Emphasis Repair] → Orphan **/​* markers stripped
    ↓
[Code-Fence-Aware Markdown Fix] → Double colons, unclosed bold (skips code blocks)
    ↓
[Mermaid Block Sanitization] → Type detection, header validation, syntax repair
    ↓
[Prompt Leak Stripping] → Removes accidental system prompt leakage
    ↓
[Structural Integrity Validator] → Final assertion layer (see below)
    ↓
Clean Markdown → UI / History Save
```

### Structural Integrity Validator (Final Gate)

Runs as the absolute last step before any response is saved or sent:

| Check | Validates | Auto-Repair |
|-------|-----------|-------------|
| **Balanced code fences** | Odd ` ``` ` count = unclosed block | Appends closing ` ``` ` |
| **Balanced emphasis** | Orphan `**` or `*` on non-code lines | Strips the trailing orphan marker |
| **Mermaid hygiene** | Empty blocks or missing diagram header | Replaces with HTML comment |
| **Response size limit** | > 32 KB (~8k tokens) | Truncates at sentence boundary + closes fences |

### Stream Sanitization

Streamed responses receive **per-chunk sanitization** via `_stream_sanitize_chunk()`:
- Unicode bullet normalization (•, ·, ‣, ◦ → `- `)
- Smart quote → ASCII quote conversion
- Em/en dash → hyphen
- Ellipsis character → `...`

After the full stream completes, the collected answer receives the **full formatting pipeline** before being saved to history.

### Key File
- `app/services/llm/response_postprocess.py` (~1,750 lines)

---

## Modular Prompt Architecture (NEW)

### Purpose
Composable system prompt construction using named `PolicyModule` objects. Eliminates monolithic prompt strings and enables fine-grained control per question type.

### PolicyModule System
Each module encapsulates a single behavioral contract:

| Module | Purpose |
|--------|---------|
| `RESPONSE_CONTRACT` | Core formatting rules (bullets, length, style) |
| `RESPONSE_TEMPLATE` | Explicit code-block, bold, table formatting rules |
| `CODE_QUALITY` | No mid-function truncation, single fence blocks |
| `SYSTEM_DESIGN` | Senior-engineer conversational style + Mermaid guidance |
| `COPILOT_SYSTEM` | Interview copilot identity + attribution |
| `UX_CONVERSATION` | Natural conversational tone enforcement |
| `MIRROR_MODE` | Separate policy set for mirror/comparison mode |

### Prompt Composition
```python
build_default_system_prompt(flags: PromptFlags) → str
```
Selects and concatenates relevant policies based on question type, domain, and routing intent. Mirror mode uses a completely separate module set.

### Key Files
- `app/prompts/policies.py` — PolicyModule definitions
- `app/prompts/builder.py` — Composer + PromptFlags
- `app/prompts/mirror_policies.py` — Mirror mode policies
- `app/prompts/response_plan.py` — ResponsePlan dataclass

---

## LLM Abstraction Architecture

### Provider-Agnostic Design

The system implements a unified `LLMService` abstraction that decouples business logic from specific LLM providers. This design enables:

- **Hot-swappable providers** via configuration (`settings.llm_provider`)
- **Consistent interface** for both streaming and non-streaming responses
- **Fallback handling** when providers are unavailable or misconfigured
- **Per-request provider override** for multi-tenant scenarios

### Provider Implementation

```
LLMService (app/services/chat/llm_service.py, ~2,740 lines)
    ├─ generate_answer() → Full conversation response
    ├─ stream_answer() → SSE streaming with per-chunk sanitization
    ├─ generate_text() → Single completion
    ├─ _format_response() → Full post-processing pipeline
    ├─ _structural_integrity_check() → Final assertion gate
    └─ Provider Routing:
        ├─ Groq SDK → Ultra-fast inference (llama-3.3-70b-versatile)
        └─ Gemini SDK → Large context windows (gemini-2.0-flash)
```

**Key Decision:** Groq chosen for speed (sub-second latency), Gemini for context length (1M tokens in latest models).

### Identity Guard System

All LLM prompts include a forward prompt (`CODE_FORWARD_PROMPT`) that enforces:
- **Attribution rules:** AI must never impersonate Stratax AI developers
- **Response formatting:** Structured output with sections (Answer, Code, Explanation)
- **Behavioral guidelines:** Professional tone, no disclaimers, actionable advice

## Agent Architecture (Practice Mode)

### Multi-Agent Orchestration

Practice Mode implements a **composite agent pattern** where specialized agents handle distinct responsibilities:

| Agent | Responsibility | LLM Usage |
|-------|---------------|-----------|
| **InterviewerAgent** | Question bank, sequencing, micro-feedback | Optional (Gemini direct) |
| **SpeechAnalyticsAgent** | Delivery metrics via DSP | None (pure signal processing) |
| **AdaptiveInterviewerAgent** | Answer evaluation (correctness, coverage) | Yes (JSON mode) |
| **ConversationalAgent** | Profile inference from natural language | Yes |
| **EvaluationAgent** | End-of-interview coaching report | Yes (JSON mode) |

### Why Multi-Agent?

**Separation of concerns:** Each agent has a single, well-defined purpose.  
**Independent scaling:** Can optimize/replace agents without affecting others.  
**Testability:** Each agent can be unit-tested in isolation.  
**Fallback resilience:** If LLM-based agent fails, rule-based fallbacks preserve core functionality.

### JSON Mode Enforcement

Several agents operate in **JSON mode** for structured output:

```python
AdaptiveInterviewerAgent.evaluate_answer_comprehensively()
    → Returns:
    {
      "correctness_score": 0-100,
      "technical_accuracy": "Excellent|Good|Fair|Poor",
      "key_points_covered": [...],
      "strengths": [...],
      "improvement_areas": [...],
      "actionable_suggestions": [...]
    }
```

**Challenge:** LLMs occasionally produce malformed JSON.  
**Mitigation:** Multi-stage JSON repair pipeline with regex-based fixing and manual extraction fallbacks.

## Retrieval-Augmented Generation (RAG)

### Hybrid Search Architecture

Interview Intelligence combines **lexical** and **semantic** retrieval for optimal question discovery:

```
Query: "Explain Python GIL"
    ↓
┌─────────────────────┬─────────────────────┐
│   BM25 (Lexical)    │  Semantic (Vector)  │
│                     │                     │
│ Exact match: "GIL"  │ Conceptual match:   │
│ High weight: Python │ "thread safety",    │
│                     │ "concurrency limits"│
└──────────┬──────────┴──────────┬──────────┘
           │                     │
           └──────────┬──────────┘
                      ↓
           Reciprocal Rank Fusion
                      ↓
              Top-K Results
```

### Why Hybrid?

- **BM25 strengths:** Exact terminology matches (framework names, specific APIs)
- **Semantic strengths:** Conceptual similarity, paraphrase handling, multilingual potential
- **Fusion:** Combines best of both without requiring relevance labels for training

### Optional Reranking

When `enable_reranking=true` and Cohere API key provided:

```
Initial results (BM25 + Semantic fusion)
    ↓
CohereReranker
    ↓
Neural model re-scores candidates
    ↓
Top-K refined results
```

**Benefit:** Cohere's reranker is trained on relevance data; improves precision by 15-30% in practice.

## Code Execution Sandbox

### Purpose
Safely validate code solutions for generated interview questions.

### Architecture
```
Code solution candidate
    ↓
CodeExecutionSandbox
    ├─ Judge0 API (primary)
    └─ Piston API (fallback)
    ↓
Execution result (stdout, stderr, exit code)
    ↓
Solution marked as verified/failed
```

### Safety Guarantees
- **External sandbox:** Code never executes on application server
- **Resource limits:** Judge0/Piston enforce CPU/memory/time constraints
- **Language support:** Python, JavaScript, Java, C++, and 40+ more

### Use Case
When generating questions, the system can:
1. Generate code solution via LLM
2. Execute in sandbox to verify correctness
3. Store only verified solutions in vector DB
4. **Result:** Higher quality question bank with tested solutions

## Query Expansion

### Purpose
Augment user queries with related terms and concepts to improve retrieval recall.

### Flow
```
User query: "React optimization"
    ↓
QueryExpansion (LLM)
    ↓
Expanded: "React optimization, useMemo, useCallback,
           React.memo, virtual DOM, component re-rendering"
    ↓
Search with expanded query
```

**Trade-off:** Improves recall but may reduce precision. Enabled via feature flag.

<div style="page-break-after: always;"></div>

# Security Model

## Authentication & Authorization

### API Key Strategy

The system implements **optional bearer token authentication**:

```
Authorization: Bearer <API_KEY>
```

**Behavior:**
- If `settings.api_key` is **set** → Authentication required for protected endpoints
- If `settings.api_key` is **unset** → Authentication bypassed (development mode)

**Protected Endpoints:**
- `POST /api/render_mermaid` (diagram generation requires auth)
- WebSocket `ws://*/ws/stt/{session_id}` (requires key via `Sec-WebSocket-Protocol` header)

### WebSocket Authentication

WebSocket connections cannot use standard HTTP headers, so the API key is passed via the WebSocket subprotocol:

```javascript
new WebSocket('ws://host/ws/stt/session123', ['<API_KEY>'])
```

Server validates via `websocket_verify_api_key` function.

### User Identification

The system extracts `user_id` from multiple sources (priority order):

1. **Header:** `X-User-ID`
2. **Bearer Token:** Extracted from `Authorization` header (JWT is verified for authenticated routes; legacy/non-auth flows may still treat the token as a caller-provided identifier)
3. **Query Parameter:** `?user_id=...`
4. **Cookie:** `user_id` cookie

**User ID Purpose:**
- Session isolation (each user has separate session directory)
- History segregation
- Audit logging

### Privacy & Data Use

Practice learning uses aggregated, anonymized metrics only; derived signals are used to improve feedback quality over time, and no raw audio, transcripts, or personally identifiable information are retained for learning purposes.

## Data Isolation

### Per-User Storage

All user data is isolated by `user_id`:

*Session persistence depends on configuration:*
- **DB-backed (default for Postgres):** sessions stored in the SQL DB
- **File-backed (common in SQLite/dev):** sessions stored under `data/sessions/{user_id}/`

```
data/
├── sessions/
│   ├── user123/
│   │   ├── session-abc.json
│   │   └── session-def.json
│   └── user456/
│       └── session-xyz.json
```

**Guarantee:** Users cannot access other users' sessions or history.

### Session Manager Design

- **In-memory cache per user:** `_user_managers` dictionary keyed by `user_id`
- **Persistence:** DB-backed by default for Postgres, or file-backed under `data/sessions/{user_id}/` when `STRATAX_SESSION_STORE=file`
- **Concurrency protection:** File locks prevent write conflicts

## CORS Configuration

**Current Setting:** Fully permissive

```python
allow_origins=["*"]
allow_methods=["*"]
allow_headers=["*"]
```

**Production Recommendation:** Restrict `allow_origins` to specific domains:

```python
allow_origins=[
    "https://app.stratax.ai",
    "https://admin.stratax.ai"
]
```

## Known Security Gaps

### 1. Mixed identity mechanisms (JWT vs. "user_id" extraction)

**Current State (as implemented):**

- JWT verification **is implemented** for authenticated routes using `app/auth.py` (FastAPI `HTTPBearer` + JWT decode).
- Separately, general request flows still use a convenience middleware that derives `request.state.user_id` from headers/cookies/query params.
    - That middleware may treat `Authorization: Bearer <token>` as a user_id when present (JWT decode is not applied there).

**Risk:** In multi-tenant deployments, using an unverified bearer token as an identifier can allow spoofing of `user_id` in non-authenticated flows.

**Recommendation:** Unify identity: prefer deriving `user_id` from validated JWT (when provided), and treat other mechanisms as guest-only identifiers.

### 2. Rate limiting is in-memory (not distributed)

**Current State:** A tier-based in-memory sliding-window limiter meters expensive LLM-backed endpoints and returns `X-RateLimit-*` headers; demo sessions have stricter caps.

**Risk:** Limits are not shared across instances/workers.

**Recommendation:** For production horizontal scaling, move rate limiting to Redis (or another shared store) and keep the same header contract.

### 3. WebSocket STT is intentionally deferred

**Current State:** The WebSocket STT endpoint exists, but the streaming transcription path is intentionally deferred (not used as the primary production STT).

**Recommendation:** Practice Mode uses faster-whisper via `LocalSTTService` (production-ready path). If you need WebSocket streaming STT, integrate a real provider (Deepgram/AssemblyAI/etc.) or implement a faster-whisper streaming adapter with backpressure + input validation.

### 4. API Keys in Configuration

**Current State:** LLM provider keys stored in `.env` file.

**Risk:** Keys could be committed to version control or exposed in logs.

**Recommendation:** Use secrets management (AWS Secrets Manager, HashiCorp Vault, Azure Key Vault).

## Audit Logging

### JSONL Audit Trail

When enabled (`settings.analytics_path`), the system logs:
- User ID
- Session ID
- Question text
- Answer text
- Timestamp
- LLM provider used

**Storage Format:** JSONL (JSON Lines) for easy parsing and streaming ingestion.

**Use Case:** Compliance, usage analytics, debugging, LLM cost tracking.

<div style="page-break-after: always;"></div>

# Deployment & Operations

## Production Readiness (short)

- Neon Postgres used for production persistence
- Docker Postgres is dev-only (profile-based)
- Multi-LLM fallback routing implemented
- Email verification & password reset enabled
- Rate limiting & demo gating enforced

Stratax is intentionally not optimized for large-scale competitive coding or leaderboard-driven learning.

## Docker Configuration

### Container Architecture

**Base Image:** `python:3.11-slim`  
**Exposed Port:** 7860  
**Entry Command:** `uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers ${UVICORN_WORKERS}`

**Worker Notes**  
- Single-worker is safest when using a file-backed local Qdrant store.
- With docker-compose (Qdrant runs as a separate service), multi-worker is supported.

### Health Check

The container exposes operational health endpoints:

```bash
GET /health
GET /health/ready
GET /health/live
```

**Response (example shape):**
```json
{
    "status": "healthy",
    "timestamp": "2026-01-01T00:00:00.000000",
    "app_name": "...",
    "environment": "...",
    "version": "1.0.0",
    "checks": {
        "database": {"status": "healthy", "latency_ms": 12.34},
        "llm_service": {"status": "configured", "provider": "gemini"},
        "vector_db": {"status": "disabled"}
    }
}
```

Note: a lightweight legacy endpoint is also available at `GET /health/simple`.

### Volume Mounts

```yaml
volumes:
    - ./app:/app/app                    # Source code (dev hot-reload)
    - ./data/history:/app/data/history  # History + audit artifacts
    - ./data/models:/app/data/models    # Embedding models
    - ./data/curated:/app/data/curated  # Curated question sets
    - ./data/vector_db:/app/data/vector_db
```

**Persistence Guarantee:** All critical state is preserved across container restarts.

## docker-compose Orchestration

### Multi-Service Setup

```yaml
services:
  web:
    build: .
    ports:
      - "7860:7860"
        env_file:
            - .env
        environment:
            - KROKI_URL=http://kroki:8000/mermaid/svg
            - DATABASE_URL=${DATABASE_URL_DOCKER:-${DATABASE_URL:-postgresql+psycopg://stratax:stratax@postgres:5432/stratax}}
            - QDRANT_URL=${QDRANT_URL:-http://qdrant:6333}
            - REDIS_URL=${REDIS_URL:-redis://redis:6379/0}
            - UVICORN_WORKERS=${UVICORN_WORKERS:-2}
        depends_on:
            - qdrant
            - redis
            - postgres

    kroki:
        image: yuzutech/kroki:latest
        ports:
            - "8001:8000"

    postgres:
        image: postgres:16
        ports:
            - "5433:5432"

    qdrant:
        image: qdrant/qdrant:latest
        ports:
            - "6333:6333"

    redis:
        image: redis:7-alpine
        ports:
            - "6379:6379"
```

See `docker-compose.yml` for the full configuration (volumes, restart policies, healthchecks).

**Kroki Integration:** Optional Mermaid diagram rendering service. If unavailable, diagrams return text-only Mermaid syntax.

## Environment Configuration

### Critical Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `GROQ_API_KEY` | Groq LLM access | Yes (if using Groq) |
| `GEMINI_API_KEY` | Google Gemini access | Yes (if using Gemini) |
| `LLM_PROVIDER` | Provider selection: `groq` or `gemini` (default: `gemini`) | No |
| `API_KEY` | Optional bearer auth for protected endpoints | No |
| `REQUIRE_USER_API_KEY` | Strict BYOK: require client-provided provider keys (no server-key fallback) | No (recommended for large rollouts) |
| `PRACTICE_MODE_ENABLED` | Enable/disable Practice Mode | No (default: true) |
| `JWT_SECRET_KEY` | JWT signing secret for `/auth/*` | Yes (in production) |
| `COOKIE_SECRET` | Cookie signing secret (OAuth state + cookies) | Yes (in production) |
| `STRATAX_SECRETS_ENCRYPTION_KEY` | Encrypt/decrypt user-stored provider keys (`enc:` prefix) | Yes (recommended; required for storing keys in production) |
| `GOOGLE_CLIENT_ID` | Google OAuth | No |
| `GOOGLE_CLIENT_SECRET` | Google OAuth | No |
| `BACKEND_BASE_URL` | OAuth redirect base (backend) | No |
| `FRONTEND_URL` | OAuth redirect base (frontend) | No |
| `ENABLE_DEMO_KEY_POOL` | Allow demo users to consume Stratax demo key pool | No |
| `STRATAX_DEMO_API_KEYS` | Pool of Groq keys for demo traffic | No |
| `EMBEDDING_CONCURRENCY_LIMIT` | Optional per-worker cap for concurrent embedding/vector-search offloads | No |

### Optional Integrations

| Variable | Purpose |
|----------|---------|
| `SERPER_API_KEY` | Web search augmentation |
| `COHERE_API_KEY` | Neural reranking |
| `JUDGE0_API_KEY` | Code execution sandbox |
| `GITHUB_TOKEN` | GitHub repository search |
| `KROKI_URL` | Diagram rendering service |
| `QDRANT_URL` | Qdrant as a service (required for multi-worker scaling) |
| `REDIS_URL` | Distributed rate limiting / cross-worker coordination |
| `SENTRY_DSN` | Sentry error tracking endpoint |
| `APP_VERSION` | Application version tag sent to Sentry |
| `QDRANT_API_KEY` | API key for managed Qdrant Cloud |
| `QDRANT_GRPC_PORT` | gRPC port for Qdrant (default: 6334) |
| `QDRANT_PREFER_GRPC` | Use gRPC transport for Qdrant (default: true) |
| `EDGE_TTS_VOICE` | Practice mode TTS voice name |
| `EDGE_TTS_RATE` | Practice mode TTS speech rate |

### Feature Flags

```bash
ENABLE_HYBRID_SEARCH=true
ENABLE_RERANKING=true
ENABLE_CODE_EXECUTION=true
ENABLE_QUERY_EXPANSION=false
ENABLE_STREAMING=true
ENABLE_PRACTICE_LEARNING=false
ENABLE_LANGGRAPH_PRACTICE=false
```

### Demo-mode cost controls (optional)

The backend can operate a public demo mode that is cost-capped by:

- Strict per-session demo quotas (minutes-based window)
- Optional global daily demo request limit
- Optional demo key pool with a safety gate to prevent burning demo keys in development

See `app/config.py` and `app/middleware/rate_limit.py`.

## Monitoring & Observability

### Health Endpoints

- **Primary:** `GET /health` (application status)
- **Identity Check:** `GET /api/identity-check` (LLM identity guard diagnostic)
- **Mock Interview:** `GET /api/mock-interview/health`
- **Intelligence:** `GET /api/intelligence/health`

### Logging Strategy

**Default:** Python logging with INFO level  
**Debug Mode:** Set `LOG_LEVEL=DEBUG` for verbose output

**Structured Logs:** Agent actions, LLM calls, and search operations log structured JSON for parsing.

### Resource Monitoring

**Key Metrics to Track:**
- **Memory:** Embedding models + Qdrant indices consume 2-4GB
- **Disk:** Growing with user sessions, audio files, vector DB
- **LLM API Quotas:** Track calls/day to Groq and Gemini
- **Audio Storage:** Practice audio files accumulate (~1MB per answer)

<div style="page-break-after: always;"></div>

# Performance & Scalability

## Current Architecture Constraints

### 1. Single-Worker Limitation

**Constraint:** `uvicorn --workers 1` due to Qdrant local file lock.

**Impact:**
- Cannot scale horizontally within a single instance
- Request throughput limited to single-process concurrency
- Long-running LLM calls can block other requests

**Mitigation Implemented:**
- Async FastAPI endpoints minimize blocking
- Shared Qdrant client in lifespan prevents lock conflicts
- LLM calls and selected heavy operations use `asyncio.to_thread` to avoid blocking the event loop
- Interview Intelligence embeddings (`SentenceTransformer.encode`) and synchronous Qdrant vector search are offloaded to threads to keep request handling responsive
- Ranking uses batched embeddings (single `encode([query] + candidates)` call) to reduce CPU work vs per-item embedding
- Optional burst protection: `EMBEDDING_CONCURRENCY_LIMIT` can cap concurrent offloaded embedding/search work per worker (default unlimited)

**Future Path:**
- Migrate to **Qdrant Cloud** or **self-hosted Qdrant server** (removes file lock)
- Enable multi-worker deployment with shared vector DB connection

## API key policy (BYOK) and operational scaling

- When `REQUIRE_USER_API_KEY=true`, key-consuming endpoints require a client-provided provider key (typically via `X-API-Key` / `X-Gemini-Key`, or in limited cases `Authorization: Bearer <provider-key>`).
- `Authorization` may contain JWTs; the backend only treats it as a provider key when it matches known provider key shapes to avoid ambiguity.
- For multi-worker deployments, set `QDRANT_URL` (shared vector DB) and `REDIS_URL` (shared rate limiting). Without these, workers behave like isolated instances.

### 2. File-Based Persistence

**Current Design:** JSON/JSONL files for sessions, history, mock interviews.

**Limitations:**
- No ACID transactions
- Manual concurrency control via file locks
- Backup/rotation requires custom scripting
- Search performance degrades with large history

**When to Migrate:**
- \> 10,000 active users
- Require distributed deployment
- Need advanced query capabilities (e.g., "find all sessions mentioning X")

**Migration Path:** PostgreSQL or MongoDB for structured data, keep Qdrant for vectors.

### 3. In-Memory Caching

**Cache Locations:**
- Code evaluation results (per-process memory)
- Session manager user caches (`_user_managers` dict)
- No cross-instance cache sharing

**Impact:** Cache misses on every request if using multiple instances or container restarts.

**Solution:** Introduce Redis for shared caching layer.

## Optimization Strategies Implemented

### 1. Embedding Model Caching

**Approach:** SentenceTransformer model loaded once at startup and reused.

**Benefit:** Avoid ~2-second model load per request.

**Implementation:** Singleton pattern in `InterviewIntelligenceService` initialization.

### 2. Audio Processing Efficiency

**faster-whisper Configuration:**
```python
model = WhisperModel(
    model_size_or_path="base",  # Configurable: tiny, base, small, medium
    device="cpu",
    compute_type="int8"          # Quantization for speed
)
```

**Trade-off:** Accuracy vs. speed. `base` model provides good balance.

### 3. Vector Search Performance

**Qdrant HNSW Index:** Approximate nearest neighbor search with sub-linear time complexity.

**Configuration:**
```python
distance = Distance.COSINE
vectors_config = VectorParams(
    size=384,              # all-MiniLM-L6-v2 embedding size
    distance=Distance.COSINE
)
```

**Performance:** <100ms for searches in databases with <100k vectors.

## Resource Requirements

### Memory Profile

| Component | Memory Usage |
|-----------|--------------|
| Base Python + FastAPI | ~200MB |
| sentence-transformers model | ~500MB |
| faster-whisper model (base) | ~400MB |
| Qdrant index (10k questions) | ~500MB |
| **Total (typical)** | **~2-3GB** |

### Disk Growth

| Data Type | Growth Rate |
|-----------|-------------|
| User sessions | ~5KB per session |
| Audio recordings | ~1MB per practice answer |
| Vector DB | ~2KB per question (embedding + metadata) |
| Model cache | ~2GB (one-time download) |

### CPU Considerations

**Heavy Operations:**
- Audio transcription (faster-whisper): 0.1-1s per 10s audio
- Embedding generation: 50-200ms per query
- LLM calls: Variable (Groq: <1s, Gemini: 1-5s)

**Recommendation:** 2-4 CPU cores for production deployment.

## Scaling Roadmap

### Phase 1: Current (Up to 1,000 concurrent users)
- ✅ Single container
- ✅ File-based persistence
- ✅ Local Qdrant

### Phase 2: Horizontal Scaling (1,000 - 10,000 users)
- 🔄 Qdrant Cloud migration
- 🔄 Multi-worker FastAPI with `--workers N`
- 🔄 Redis for shared caching
- 🔄 Database (PostgreSQL) for sessions

### Phase 3: Distributed (10,000+ users)
- 🔄 Microservices split (Q&A, Practice, Intelligence as separate services)
- 🔄 Message queue for async jobs (Celery + RabbitMQ)
- 🔄 Object storage (S3) for audio files
- 🔄 CDN for TTS audio delivery

<div style="page-break-after: always;"></div>

# Known Issues & Recommendations

## Production-Grade Transparency

*This section intentionally documents current limitations to enable informed deployment decisions.*

### Forward-looking hardening checklist

- **JWT hardening (header separation + consistency)**: keep JWT auth in `Authorization` and move user-provided LLM keys to a dedicated header (to avoid ambiguity); consistently use the JWT `sub` as the canonical user identifier.
- **Redis rate limiting (distributed quotas)**: enable the existing Redis-backed limiter by setting `REDIS_URL` so quotas are shared across workers/instances and resilient to restarts.
- **Multi-worker scaling**: migrate file-based persistence to Postgres/object storage and run Qdrant as a separate service/managed instance to avoid file-lock conflicts.
- **Production STT**: WebSocket STT is intentionally deferred; if you need streaming STT, integrate a real provider (Deepgram/AssemblyAI/etc.) or implement a faster-whisper streaming adapter.

## Critical Issues

### 1. Port Mismatch in Docker Healthcheck

**Status:** Resolved — `docker-compose.yml` healthcheck targets `http://localhost:7860/health`.

---

### 2. WebSocket STT is intentionally deferred

**Status:** WebSocket STT is intentionally deferred; Practice Mode uses faster-whisper (production-ready).

**Current Behavior:** `STTService.stream_transcribe()` yields placeholder tokens (`"..."` or `"(audio)"`) rather than real transcriptions.

**Impact:** The WebSocket STT endpoint is not intended for production streaming transcription.

**Recommendation:** Keep WebSocket STT disabled/unused in production until a real streaming STT provider (Deepgram/AssemblyAI/etc.) or a faster-whisper streaming adapter is implemented.

**Priority:** Medium — Practice Mode unaffected; only impacts WebSocket-based UX.

---

### 3. JWT verification is implemented (with legacy identity fallbacks)

**Current State (as implemented):** JWT verification is implemented for authenticated routes via `app/auth.py` (FastAPI `HTTPBearer` + JWT decode).

**Important nuance:** A legacy/convenience `user_id` extraction path still exists for non-auth / demo / guest flows, where request identity may be derived from headers/cookies/query params.

**Risk:** In multi-tenant deployments, treating an unverified bearer token (or other caller-controlled identifier) as a stable `user_id` in non-auth flows can allow spoofing.

**Recommendation:** Keep JWT auth in `Authorization` and move user-provided LLM keys / guest identifiers to dedicated headers. Prefer the validated JWT `sub` as the canonical user identifier when present.

---

### 4. Qdrant File Lock Prevents Multi-Worker Scaling

**Issue:** Local Qdrant storage uses file lock; only one process can access at a time.

**Impact:** Cannot use `uvicorn --workers N` for load distribution.

**Current Mitigation:** Shared Qdrant client initialized in lifespan; single worker.

**Long-Term Solution:** Migrate to Qdrant Cloud or self-hosted Qdrant server.

**Priority:** Medium — Single-worker adequate for 1,000 concurrent users with async endpoints.

---

## Design Trade-offs

### 5. Mixed Persistence Models

**Current State:**
- User sessions: Per-user JSON files
- Mock interview: Single JSON file for all sessions
- History: JSONL format
- Vector DB: Qdrant local storage

**Benefit:** Simple, no database dependency, easy debugging.

**Limitation:** 
- No ACID guarantees
- Manual backup required
- Search requires file scanning

**Recommendation:** Acceptable for <10k users; plan DB migration beyond that threshold.

**Priority:** Low (currently) — Reassess at scale.

---

### 6. In-Memory Code Evaluation Cache

**Issue:** Cache doesn't persist across restarts or workers.

**Impact:** Redundant LLM calls for identical code submissions.

**Solution:** Implement Redis-backed cache with TTL:
```python
import redis
r = redis.Redis(host='localhost', port=6379)
cache_key = f"eval:{hash(code)}"
result = r.get(cache_key)
if not result:
    result = evaluate_code(code)
    r.setex(cache_key, 3600, json.dumps(result))
```

**Priority:** Low — Optimization, not a bug.

---

## Future Enhancements

### 7. Async Background Jobs

**Gap:** Long-running operations (e.g., LLM calls >5s) block request handling.

**Recommendation:** Introduce task queue (Celery + RabbitMQ) for:
- Final evaluation report generation
- Bulk question curation
- Web search aggregation

**Priority:** Low — Async FastAPI handles most cases; needed at 10k+ concurrent requests.

---

### 8. Vector DB Backup Strategy

**Current State:** No automated backup for Qdrant indices.

**Risk:** Data loss if `data/interview_intelligence_v2/` corrupted.

**Recommendation:** Implement periodic snapshots:
```bash
# Cron job
0 2 * * * tar -czf /backups/qdrant-$(date +\%Y\%m\%d).tar.gz /app/data/interview_intelligence_v2/
```

**Priority:** Medium — Important for production.

---

## Debugging Recommendations

**For Developers:**
1. **Qdrant Lock Errors:** Ensure only one server instance; check for stale processes.
2. **Empty LLM Responses:** Verify API keys loaded correctly (`GET /health`).
3. **Audio Transcription Failures:** Check faster-whisper model downloaded to `data/models/`.
4. **Unexpected Port:** Application runs on **7860**, not 8000.

<div style="page-break-after: always;"></div>

# Onboarding Guide

## Quick Start (5 Minutes)

### Prerequisites
- **Python 3.11+**
- **4GB+ RAM** (for embedding models)
- **Groq or Gemini API key** (for LLM features)

### Installation

```bash
# 1. Navigate to project
cd <your-repo-folder>

# 2. Create virtual environment
python -m venv .venv

# 3. Activate environment
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\Activate.ps1  # Windows PowerShell

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure environment
echo "GROQ_API_KEY=your_key_here" > .env
echo "LLM_PROVIDER=groq" >> .env

# 6. Run application
uvicorn app.main:app --host 0.0.0.0 --port 7860

# 7. Verify
curl http://localhost:7860/health
```

**Expected Response:**
```json
{
    "status": "ok",
    "version": "0.1.0",
    "llm": {"provider": "gemini", "enabled": true}
}
```

---

## Developer Learning Path

### Day 1: Core Architecture
**Goal:** Understand application composition and routing.

1. **Read `app/main.py`** (100 lines)
   - Router registration
   - Middleware stack
   - Lifespan initialization

2. **Review `app/config.py`** (300 lines)
   - All settings and feature flags
   - Environment variable mapping

3. **Skim `app/schemas.py`** (500 lines)
   - Request/response contracts
   - Enum definitions

**Time:** 1-2 hours

---

### Day 2: Core Features
**Goal:** Understand Q&A and Practice Mode.

1. **Explore `app/routers/questions.py`**
   - Session creation
   - LLM invocation
   - Profile upload
   - Architecture/system design streaming

2. **Explore `app/routers/practice_mode.py`**
   - Round selection
   - Audio submission
   - Evaluation flow

3. **Review `app/services/chat/llm_service.py`**
   - Provider abstraction
   - Identity guard
   - Streaming support
   - Response post-processing delegation

4. **Review `app/prompts/policies.py`**
   - PolicyModule definitions
   - RESPONSE_CONTRACT, CODE_QUALITY, SYSTEM_DESIGN policies

**Time:** 2-3 hours

---

### Day 3: Advanced Components
**Goal:** Deep dive into AI subsystems.

1. **Study `app/services/interview/interview_intelligence_service.py`**
   - Vector search
   - Question generation
   - Hybrid retrieval

2. **Study `app/services/practice/practice_mode_service.py`**
   - Agent composition
   - Orchestration logic
   - STT/TTS integration

3. **Study `app/services/practice/adaptive_interviewer_agent.py`**
   - Comprehensive answer evaluation
   - Cross-session semantic deduplication
   - Follow-up drilling logic

4. **Review `app/services/llm/response_postprocess.py`**
   - Full formatting pipeline
   - Mermaid sanitization
   - Structural integrity validator

5. **Review `app/services/chat/ai_native_enhancements.py`**
   - Hybrid search
   - Reranking
   - Code execution

**Time:** 3-4 hours

---

### Day 4: Testing & Deployment
**Goal:** Validate understanding and deploy.

1. **Run test suite:**
   ```bash
   pytest -q                          # Full suite (~153 tests)
   pytest tests/test_structural_integrity_validator.py -v  # Validator tests
   pytest tests/evals/ -v             # Copilot contract evals
   ```

2. **Try Docker build:**
   ```bash
   docker build -t stratax-ai .
   docker run -p 7860:7860 stratax-ai
   ```

3. **Review `docker-compose.yml`**
   - Multi-service orchestration
   - Volume mounts
   - Environment passing

**Time:** 2 hours

---

## Key Files by Complexity

| File | Purpose | Complexity | Lines |
|------|---------|------------|-------|
| `app/main.py` | Application entry | ⭐ Low | ~290 |
| `app/config.py` | Settings & flags | ⭐ Low | ~625 |
| `app/middleware/auth.py` | User ID extraction | ⭐ Low | ~200 |
| `app/schemas.py` | Data contracts | ⭐⭐ Medium | ~970 |
| `app/prompts/policies.py` | PolicyModule definitions | ⭐⭐ Medium | ~350 |
| `app/prompts/builder.py` | System prompt compositor | ⭐⭐ Medium | ~200 |
| `app/routers/questions.py` | Q&A + copilot endpoints | ⭐⭐⭐ High | ~1,755 |
| `app/routers/practice_mode.py` | Practice endpoints | ⭐⭐⭐ High | ~2,134 |
| `app/services/chat/llm_service.py` | LLM abstraction + streaming | ⭐⭐⭐ High | ~2,740 |
| `app/services/llm/response_postprocess.py` | Response formatting pipeline | ⭐⭐⭐ High | ~1,750 |
| `app/services/practice/practice_mode_service.py` | Practice orchestrator | ⭐⭐⭐ High | ~1,030 |
| `app/services/practice/adaptive_interviewer_agent.py` | Adaptive eval + semantic dedup | ⭐⭐⭐⭐ Very High | ~1,810 |
| `app/services/interview/interview_intelligence_service.py` | Vector search + generation | ⭐⭐⭐⭐ Very High | ~3,160 |

---

## Common Developer Tasks

### Add a New LLM Provider
1. Add API key to `app/config.py`:
   ```python
   anthropic_api_key: str = ""
   ```
2. Extend `app/services/chat/llm_service.py`:
   ```python
   elif self.provider == "anthropic":
       import anthropic
       client = anthropic.Client(api_key)
       # ... implementation
   ```
3. Update `settings.llm_provider` validation

---

### Add a New Router
1. Create `app/routers/my_feature.py`:
   ```python
   from fastapi import APIRouter
   router = APIRouter()
   
   @router.get("/my-endpoint")
   async def my_endpoint():
       return {"status": "ok"}
   ```
2. Register in `app/main.py`:
   ```python
   from app.routers.my_feature import router as my_router
   app.include_router(my_router, prefix="/api/my-feature")
   ```

---

### Add a New Agent
1. Create `app/services/<subdirectory>/my_agent.py`:
   ```python
   class MyAgent:
       def __init__(self, config):
           self.config = config
       
       async def process(self, input):
           # Agent logic
           return result
   ```
2. Instantiate in orchestrating service (e.g., `practice_mode_service.py`):
   ```python
   self.my_agent = MyAgent(config)
   ```
3. Call in workflow:
   ```python
   result = await self.my_agent.process(data)
   ```

---

## Environment Configuration Priority

1. **System environment variables** (highest)
2. **Env files** (via `python-dotenv`)
3. **`Settings` class defaults** (lowest)

Notes
- This backend supports layered env files using `ENV_FILE` (comma-separated), e.g. `ENV_FILE=.env,.env.local`. This is useful for local overrides without editing the base `.env`.

**Example:**
```bash
# .env file
GROQ_API_KEY=abc123
LLM_PROVIDER=groq

# System override (takes precedence)
export LLM_PROVIDER=gemini
```

---

## Debugging Tips

| Issue | Solution |
|-------|----------|
| **Qdrant lock error** | Ensure single server instance; kill stale processes |
| **Empty LLM responses** | Check API keys: `cat .env \| grep API_KEY` |
| **Audio transcription fails** | Verify model downloaded: `ls data/models/` |
| **Port 8000 vs 7860 confusion** | Application runs on **7860** |
| **"Database is locked"** | Only one Qdrant client allowed; check `lifespan` initialization |

---

## Resources

- **FastAPI Documentation:** https://fastapi.tiangolo.com
- **Qdrant Documentation:** https://qdrant.tech/documentation
- **sentence-transformers:** https://www.sbert.net
- **faster-whisper:** https://github.com/guillaumekln/faster-whisper

<div style="page-break-after: always;"></div>

# Appendices

## Appendix A: Complete API Reference

Note: This appendix is a practical quick reference and is intentionally **not exhaustive**.
For the complete, code-derived endpoint table (including History, WebSockets, and all Practice progress endpoints), see `TECHNICAL_DOCUMENTATION.md`.

### Core Q&A Endpoints
- `POST /api/session` — Create new session
- `POST /api/question` — Submit question
- `POST /api/upload_profile` — Upload resume/JD
- `GET /api/sessions` — List sessions
- `DELETE /api/session/{id}` — Delete session
- `PUT /api/session/{id}/title` — Update title
- `GET /api/session/{id}/chat` — Get history

### Practice Mode Endpoints
- `GET /api/practice/rounds/available` — List rounds
- `POST /api/practice/interview/start-round` — Start round
- `POST /api/practice/interview/quick-start` — Quick start
- `POST /api/practice/proctoring/event` — Proctoring signals (events only; no media)
- `POST /api/practice/interview/submit-answer` — Submit audio
- `POST /api/practice/interview/acknowledge-feedback` — Next question
- `GET /api/practice/session/{id}/evaluation` — Final report

### Interview Intelligence Endpoints
- `GET /api/intelligence/search` — Search questions
- `POST /api/intelligence/search/enhanced` — Enhanced search
- `POST /api/intelligence/curate` — Curate to DB
- `GET /api/intelligence/companies` — List companies
- `POST /api/intelligence/code/execute` — Execute code

### Mock Interview Endpoints
- `POST /api/mock-interview/sessions/start` — Start session
- `POST /api/mock-interview/sessions/submit-answer` — Submit answer
- `GET /api/mock-interview/sessions/{id}/summary` — Get summary
- `POST /api/mock-interview/sessions/{id}/hint` — Request hint

*Full endpoint documentation: 80+ total endpoints across 11 routers.*

---

## Appendix B: Environment Variables

### Required
- `GROQ_API_KEY` or `GEMINI_API_KEY`
- `LLM_PROVIDER` (groq|gemini)

### Optional
- `API_KEY` — Bearer auth
- `SERPER_API_KEY` — Web search
- `COHERE_API_KEY` — Reranking
- `JUDGE0_API_KEY` — Code execution
- `GITHUB_TOKEN` — GitHub search
- `KROKI_URL` — Diagram rendering

### Feature Flags
- `ENABLE_HYBRID_SEARCH`
- `ENABLE_RERANKING`
- `ENABLE_CODE_EXECUTION`
- `PRACTICE_MODE_ENABLED`
- `ENABLE_PRACTICE_LEARNING`

*Full list: 50+ configurable settings in `app/config.py`*

---

## Appendix C: Persistence Paths

| Data Type | Location |
|-----------|----------|
| User sessions | SQL DB (default for Postgres) or `data/sessions/{user_id}/{session_id}.json` when `STRATAX_SESSION_STORE=file` |
| Mock interviews | `data/sessions/mock_interview_sessions.json` |
| Practice audio | `data/practice_audio/*.wav` |
| Vector DB | `data/interview_intelligence_v2/vector_db/` |
| Models cache | `data/models/` |
| History logs | Configured via `settings.analytics_path` |

---

## Appendix D: Test Coverage

**Test Files:** 80+ across `tests/` directory and root  
**Last verified:** February 2026

Key test areas:
- `tests/test_structural_integrity_validator.py` — Structural integrity validator (fences, emphasis, Mermaid, size)
- `tests/test_llm_response_bullet_normalization.py` — Bullet/emphasis formatting
- `tests/evals/test_copilot_contract_evals.py` — Copilot prompt & routing contract evals
- `tests/test_adaptive_interviewer_agent_json_parsing.py` — Agent JSON recovery
- `tests/test_auth_system.py` — JWT auth flows
- `tests/test_company_specific.py` — Targeted question generation
- `tests/test_architecture_dynamic_limits.py` — Complexity detection
- `tests/test_event_logging_question_flow.py` — Telemetry pipeline
- `tests/test_difficulty_levels.py` — Practice difficulty adaptation
- `tests/test_code_execution_endpoint.py` — Sandbox execution
- `tests/test_resume_parser.py` — Resume parsing & claim extraction
- `tests/test_mirror_ontology.py` — Mirror ontology generation & caching
- `tests/test_mirror_compare.py` — Mirror cross-attempt progress diff
- `tests/test_practice_scoring.py` — Deterministic practice scoring rubrics
- `tests/test_adaptive_pressure.py` — Adaptive difficulty adjustment
- `tests/test_dynamic_budget.py` — Token budget computation
- `tests/test_mock_interview_analytics.py` — Mock interview trajectory
- `tests/test_email_verification_and_password_reset.py` — Email auth flows
- `tests/test_rate_limiting.py` — Tier-based rate limiting
- `tests/test_domain_profile.py` — Domain profile detection
- `tests/test_interview_intelligence_requires_user_key.py` — II BYOK enforcement

*Full test suite in repository root.*

---

## Appendix E: Troubleshooting Commands

### Find Process on Port
```bash
# Linux/Mac
lsof -ti:7860 | xargs kill -9

# Windows
netstat -ano | findstr :7860
taskkill /PID <pid> /F
```

### Check Qdrant Lock
```bash
ps aux | grep uvicorn
# Kill stale processes
pkill -f uvicorn
```

### Verify API Keys
```bash
cat .env | grep -E '(GROQ|GEMINI)_API_KEY'
```

### Enable Debug Logging
```bash
export LOG_LEVEL=DEBUG
uvicorn app.main:app --log-level debug
```

---

*End of Technical Documentation*

---

**Document Metadata**
- **Total Pages:** 35+
- **Word Count:** ~10,000
- **Code Examples:** Minimal (architecture-focused)
- **Target Audience:** CTOs, Architects, Senior Engineers
- **Classification:** Technical Architecture & System Design

---

*Generated with precision from repository codebase.*  
*Stratax AI Development Team — February 2026*
