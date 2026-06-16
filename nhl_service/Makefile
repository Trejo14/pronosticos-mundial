.PHONY: all help install format lint test migrate run local shell ingest

all: help

help:
	@echo "Available commands:"
	@echo "  install    - Install dependencies (with dev group)"
	@echo "  format     - Auto-format code with ruff"
	@echo "  lint       - Lint and typecheck with ruff/mypy"
	@echo "  test       - Run pytest suite with coverage"
	@echo "  migrate    - Run Django database migrations"
	@echo "  run        - Make migrations and run local dev server"
	@echo "  shell      - Open Django shell_plus"

install:
	pip install -e ".[dev]"

format:
	ruff check --fix .
	ruff format .

lint:
	ruff check .
	mypy .

test:
	pytest tests/

migrate:
	python manage.py makemigrations
	python manage.py migrate

run: migrate
	python manage.py runserver 0.0.0.0:8001

shell:
	python manage.py shell
