# Contributing to maestro-memory

## Development Setup

```bash
git clone https://github.com/maestro-ai-stack/maestro-memory.git
cd maestro-memory
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,local]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Running Benchmarks

Built-in retrieval benchmark (fast, no API keys):
```bash
python evals/run_retrieval_eval.py
```

LongMemEval benchmark (requires dataset download):
```bash
# Download data first
cd ~/LongMemEval/data/
curl -sL https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json -o longmemeval_s_cleaned.json

# Run retrieval eval
python evals/run_longmemeval.py --data s --limit 50
```

## Pull Request Process

1. Fork the repo and create a feature branch.
2. Make your changes with clear, atomic commits.
3. Ensure all tests pass and ruff reports no issues.
4. Open a PR with a description of what changed and why.

## Reporting Bugs

Open an issue with:
- Python version and OS
- Minimal reproduction steps
- Expected vs actual behavior
- Error traceback if applicable

## Architecture

See [docs/TECHNICAL.md](docs/TECHNICAL.md) for technical principles and [docs/ARCHITECTURE-v2.md](docs/ARCHITECTURE-v2.md) for the v2 roadmap.
