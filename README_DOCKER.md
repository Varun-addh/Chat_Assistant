# Docker: Build and Run

This document explains how to build and run the InterviewAst backend using Docker (development and production recommendations).

Prerequisites
- Docker installed (Windows: Docker Desktop)
- Optional: docker-compose

Quickstart (development)
1. Copy the example env file and fill keys:
   - Copy `.env.example` to `.env` and add your API keys.

2. Build and run with docker-compose (dev, mounts source for live reload):

   On PowerShell (pwsh.exe):

   # Build and start
   docker-compose build --pull
   docker-compose up --detach

   # View logs
   docker-compose logs -f

   # Stop and remove
   docker-compose down

Running a single container (docker)

  docker build -t interviewast:dev .
  docker run --rm -it -p 8000:8000 --env-file .env -v ${PWD}:/app interviewast:dev

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

Persistent data
- If you need to persist data (vector DB, session files, embeddings cache), mount the `data/` directory as a volume in production.

Troubleshooting
- If the container fails to start, run it interactively to see logs:
  docker run --rm -it -p 8000:8000 --env-file .env interviewast:dev

- Check logs from docker-compose: docker-compose logs -f

