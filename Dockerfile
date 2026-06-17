# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY espn_service/pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY espn_service/ .

EXPOSE $PORT

CMD python manage.py collectstatic --noinput --clear 2>&1 || true; python manage.py migrate --noinput 2>&1 || true; gunicorn config.wsgi:application --bind 0.0.0.0:${PORT:-8000} --workers 2 --threads 2 --access-logfile - --error-logfile -
