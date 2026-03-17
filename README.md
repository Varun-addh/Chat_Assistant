---
title: Stratax AI - Interview Assistant
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Stratax AI - Interview Intelligence Platform

An advanced AI-powered interview preparation platform featuring:

- 🤖 **AI Copilot Chat Assistant** - Get expert answers to technical and behavioral interview questions
- 📊 **Multi-View System Design** - Generate comprehensive architecture diagrams with 5 focused views
- 🎤 **Mock Interview Practice** - Practice with AI interviewer and get real-time feedback
- 🔍 **Search Intelligence** - Find curated interview questions from top companies
- 📈 **Live Practice Mode** - Real-time speech-to-text practice with analytics

## Features

### 1. AI Copilot Chat
- Context-aware responses using advanced LLMs (Gemini/Groq)
- Intelligent API provider selection based on configured keys
- System design architecture generation (single-view and multi-view)
- Code evaluation and explanation

### 2. Interview Intelligence
- Curated questions from FAANG+ companies
- Difficulty-based filtering (Easy, Medium, Hard)
- Topic-specific question generation
- Company-specific interview prep

### 3. Mock Interview
- Simulated interview sessions
- Real-time feedback and scoring
- Adaptive difficulty based on performance
- Detailed performance analytics

### 4. Live Practice Mode
- Speech-to-text transcription
- Real-time answer evaluation
- Speaking pace and clarity analysis
- Offline TTS for interviewer questions

## Technology Stack

 **Backend**: FastAPI (Python 3.11+)
- **AI/LLM**: Google Gemini API, Groq API
- **Vector DB**: Qdrant (for semantic search)
- **Embeddings**: SentenceTransformers
 **Architecture**: Monolithic FastAPI service with Docker/docker-compose-managed dependencies (Postgres, Qdrant, Redis, optional Kroki)

## Configuration

The application supports flexible API key configuration:

- **Only Groq API key**: All features use Groq
- **Only Gemini API key**: All features use Gemini
- **Both API keys**: AI Copilot uses Gemini, other features use Groq

Set your API keys in the Bridge Settings or via environment variables.

Environment files (recommended workflow)

- Copy `.env.example` to `.env` and fill in your secrets (this file is ignored by git).
- Put machine-specific overrides in `.env.local` (also ignored).
- Select layered env files without editing `.env` using `ENV_FILE`, e.g.:
	- `.env,.env.local` for local dev
	- `.env,.env.docker` for docker-like overrides

## Development

For reproducible installs, prefer `pip install -r requirements.lock`. Use `requirements.txt` only when intentionally updating dependency versions.

See [README_DOCKER.md](README_DOCKER.md) for Docker deployment instructions.

## Scaling on Hugging Face Spaces (CPU Basic)

Hugging Face Spaces typically runs your Docker Space as a **single container instance** (no automatic horizontal autoscaling). On CPU Basic (2 vCPU / 16 GB), performance is mostly about avoiding event-loop blocking, keeping CPU-bound work off the main loop, and being careful with multi-process settings.

Recommended defaults

- Keep `UVICORN_WORKERS=1` on CPU Basic unless you have moved shared state to external services.
- If you enable strict BYOK for a university rollout, set `REQUIRE_USER_API_KEY=true` so the backend never falls back to server keys.

When (and how) to increase workers

- Increasing `UVICORN_WORKERS` can help with I/O concurrency, but it also multiplies memory usage and requires shared backends.
- If you run multiple workers, you should set:
	- `REDIS_URL` (rate limiting + any cross-worker coordination; without Redis, each worker rate-limits independently)
	- `QDRANT_URL` (so all workers talk to the same Qdrant service; local/on-disk Qdrant mode is not safe to “share” across workers)
	- A real external `DATABASE_URL` (Postgres/Neon) for persistence

Practical notes

- CPU-heavy features (embeddings, reranking, some audio workloads) will still be bounded by your 2 vCPU. If Practice Mode STT/TTS is heavily used, consider upgrading the Space or moving those workloads to separate services.
- Optional burst protection: set `EMBEDDING_CONCURRENCY_LIMIT` (e.g. `2` or `4`) to cap concurrent embedding/vector-search offloads per worker. Default is unlimited.
- If you don’t need Interview Intelligence or Practice Mode on Spaces, you can reduce startup/load using flags like `FAST_STARTUP=true` or `DISABLE_INTERVIEW_INTELLIGENCE=true`.

## License

MIT License - See LICENSE file for details

---

**Developed by Varun Bikkumalla**
