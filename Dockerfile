# ─────────────────────────────────────────────
# Stage 1: Builder
# Multi-stage build: keeps final image lean and avoids shipping build tools.
# Mention in interview: "Multi-stage reduces attack surface and image size."
# ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps (not shipped in final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ─────────────────────────────────────────────
# Stage 2: Runtime
# ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user (security best practice — standard ASML/enterprise requirement)
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy only installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/
COPY prompts/ ./prompts/

# GCS credential mount point (injected via K8s secret, not baked in)
ENV GOOGLE_APPLICATION_CREDENTIALS=/var/secrets/google/key.json
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Drop to non-root
USER appuser

EXPOSE 8080

# Gunicorn + Uvicorn workers:
# - Gunicorn manages worker processes (handles concurrency)
# - Uvicorn workers handle async event loop (non-blocking LLM streams)
# Mention in interview: "This setup prevents long LLM responses from blocking
# other requests — critical when Nginx proxies streaming SSE responses."
CMD ["gunicorn", "app.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8080", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-"]
