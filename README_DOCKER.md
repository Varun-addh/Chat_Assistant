# Docker: Build and Run

This document explains how to build and run the Stratax AI backend using Docker (development and production recommendations).

Prerequisites
- Docker installed (Windows: Docker Desktop)
- Optional: docker-compose

Ports
- The Docker image in this repo runs Uvicorn on port **7860** (see `Dockerfile`).
- Local development without Docker commonly uses **8000**.

Quickstart (development)
1. Copy the example env file and fill keys:
   - Copy `.env.example` to `.env` and add your API keys.

  Tip (recommended): keep `.env` as your base file and put machine-specific overrides in `.env.local`.
  The backend supports loading multiple env files via `ENV_FILE` (comma-separated).

2. Build and run with docker-compose (dev, mounts source for live reload):

   On PowerShell (pwsh.exe):

   # Build and start
  docker compose build --pull
  docker compose up --detach

   # View logs
  docker compose logs -f

   # Stop and remove
  docker compose down

  The compose stack brings up:
  - `web` (FastAPI)
  - `postgres` (session/history persistence when DATABASE_URL is Postgres)
  - `qdrant` (vector DB as a service; required for multi-worker)
  - `kroki` (local Mermaid renderer)

  After startup, the API will be available at:
  - http://localhost:7860

Running a single container (docker)

  docker build -t interviewast:dev .
  docker run --rm -it -p 7860:7860 --env-file .env -v ${PWD}:/app interviewast:dev

Environment selection without editing `.env`

- Local uvicorn (example): set `ENV_FILE=.env,.env.local`
- Docker compose (example): you can optionally add overrides using a second env file.
  Newer Compose supports multiple `--env-file` flags.
  If yours does not, keep overrides in `.env` or use shell env vars.

Production recommendations
- Do not mount the host source into the container.
- Remove `--reload` from the Uvicorn command and use multiple workers.
- Use a process manager (systemd, Kubernetes) or `docker-compose` with restart policies.
- Example production CMD (Dockerfile change):

  CMD ["/usr/local/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

Health checks
- docker-compose.yml includes a simple healthcheck which pings `/health`.
- Ensure `app.main` exposes a `/health` endpoint; if not present, change the healthcheck or add the endpoint.

Environment variables
- Required: API keys like SERPER_API_KEY must be set in the container (via `.env` or directly in the environment).

Qdrant + multi-worker
- If you run multiple Uvicorn workers, set `QDRANT_URL` so all workers share the same Qdrant service.
- `docker-compose.yml` defaults `QDRANT_URL` to `http://qdrant:6333` for the `web` container.

Persistent data
- If you need to persist data (vector DB, session files, embeddings cache), mount the `data/` directory as a volume in production.

Troubleshooting
- If the container fails to start, run it interactively to see logs:
  docker run --rm -it -p 7860:7860 --env-file .env interviewast:dev

- Check logs from docker-compose: docker compose logs -f

