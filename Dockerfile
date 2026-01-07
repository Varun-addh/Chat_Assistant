FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    ffmpeg \
    libsndfile1 \
    pkg-config \
    libxml2-dev \
    libxslt-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip early
RUN pip install --upgrade pip setuptools wheel

# -------- CRITICAL LAYER CACHING --------
# Install heavy ML deps first
RUN pip install \
    torch==2.2.2 \
    torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Copy and install rest
COPY requirements.txt .
RUN pip install -r requirements.txt

# Safety net: ensure critical runtime deps are present (HF build caches can be tricky).
RUN pip install --no-cache-dir email-validator \
    && python -c "import email_validator; import jwt; print('deps ok')"

# Copy application
COPY . .

# Runtime folders & permissions
RUN mkdir -p /app/data/{history,models,curated,vector_db,interview_intelligence_v2,qdrant} \
    && useradd -m -u 1000 user \
    && chown -R user:user /app

USER user
ENV PATH="/home/user/.local/bin:$PATH"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
  CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
