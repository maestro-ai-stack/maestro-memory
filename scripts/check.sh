#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

"$PYTHON" scripts/check_local_only.py
"$PYTHON" -m ruff check src tests evals scripts/check_local_only.py
"$PYTHON" -m pytest -q
rm -rf dist
"$PYTHON" -m build
"$PYTHON" -m twine check dist/*
