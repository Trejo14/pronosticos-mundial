# Base image
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir build && \
    pip wheel --no-deps --requirement <(python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))") -w /wheels

# Final stage
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy application code
COPY . .

# Run as non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE="config.settings.production"

EXPOSE 8000

CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "4", "--access-logfile", "-", "--error-logfile", "-"]
