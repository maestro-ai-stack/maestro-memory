.PHONY: help gate gate-baseline gate-live test lint export

PY ?= python3
GOLD ?= $(HOME)/.maestro/backups/nerve-export-20260818/gold_queries.jsonl
# Versioned deliberately: this is the gate's reference point, not a run artifact,
# so it lives outside the gitignored evals/results/ directory.
BASELINE ?= evals/gate_baseline.json
SCRATCH_DB ?= /tmp/mmem-gate.db
LIVE_DB ?= $(HOME)/.maestro/memory/default/mem.db

help:
	@echo "make gate           Score the live store against the real labelled query set"
	@echo "make gate-baseline  Re-derive the as-served baseline from the gold log"
	@echo "make export         Re-export the mnerve Postgres store to JSONL"
	@echo "make test           Run the test suite"

## Reproduce the pre-migration baseline from the recorded ranking.
## This is the harness self-check: it must match the numbers computed
## independently in SQL against the mnerve Postgres store
## (P@1 0.5952, Hit@3 0.8175, Hit@5 0.8730, MRR 0.7215 over 126 labelled queries).
gate-baseline:
	$(PY) evals/gate.py --recorded --gold $(GOLD) --save $(BASELINE)

## The gate. Runs against a scratch COPY of the store: Memory.search()
## writes a serving_logs row per query, so gating the live DB would inject
## 155 synthetic rows into the ranking training data on every run.
gate:
	@cp $(LIVE_DB) $(SCRATCH_DB)
	$(PY) evals/gate.py --engine mmem --db $(SCRATCH_DB) --gold $(GOLD) --baseline $(BASELINE)
	@rm -f $(SCRATCH_DB)

## Gate the live store in place. Only for deliberate one-off inspection.
gate-live:
	$(PY) evals/gate.py --engine mmem --db $(LIVE_DB) --gold $(GOLD) --baseline $(BASELINE)

export:
	./migrate/export_from_nerve.sh

test:
	$(PY) -m pytest tests/ -q

lint:
	ruff check src/ evals/
