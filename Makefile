.PHONY: help check test lint

PY ?= python3

help:
	@echo "make check  Run the complete local release gate"
	@echo "make test   Run the test suite"
	@echo "make lint   Run Ruff over runtime and tests"

check:
	./scripts/check.sh

test:
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m ruff check src/ tests/ evals/ scripts/check_local_only.py
