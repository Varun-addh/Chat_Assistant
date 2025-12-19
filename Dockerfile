# Optimized for deployment
FROM python:3.12-slim

# Avoid .pyc files, ensure clean logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    pkg-config \
    libxml2-dev \
    libxslt-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt /app/requirements.txt

# Install dependencies using cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

# Ensure data directories exist and have correct permissions for appuser.
# We create all potential subdirectories used by services to avoid PermissionDenied on mkdir.
RUN mkdir -p /app/data/history \
             /app/data/models \
             /app/data/curated \
             /app/data/vector_db \
             /app/data/interview_intelligence_v2 \
             /app/data/qdrant \
    && chmod -R 777 /app/data

# Create and use non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production server command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
