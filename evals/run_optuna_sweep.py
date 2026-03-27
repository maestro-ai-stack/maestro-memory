#!/usr/bin/env python3
"""Optuna parameter sweep for mmem LongMemEval retrieval.

Searches for optimal retrieval parameters using Bayesian optimization.
Objective: maximize session recall on 50 stratified LongMemEval_S questions.
Zero cost — no API calls, purely local computation.

Usage:
  pip install optuna
  python evals/run_optuna_sweep.py --n-trials 100
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import optuna
except ImportError:
    print("pip install optuna")
    sys.exit(1)

from maestro_memory import Memory

LONGMEMEVAL_DIR = Path.home() / "maestro/projects/LongMemEval/data"
RESULTS_DIR = Path(__file__).parent / "results"


def load_sample() -> list[dict]:
    """Load 50 stratified questions (same sample as autoresearch loop)."""
    with open(LONGMEMEVAL_DIR / "longmemeval_s_cleaned.json") as f:
        data = json.load(f)
    by_type = defaultdict(list)
    for item in data:
        by_type[item["question_type"]].append(item)
    sampled = []
    for qtype, items in by_type.items():
        n = max(5, len(items) * 50 // 500)
        sampled.extend(items[:n])
    return sampled[:50]


async def eval_with_params(
    data: list[dict],
    trunc_limit: int,
    top_k: int,
    chunk_mode: str,  # "whole", "pairs", "pairs+summary"
    summary_importance: float,
) -> float:
    """Evaluate session recall with given params."""
    total_recall = 0.0
    for item in data:
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = Memory(path=Path(tmpdir) / "eval.db")
            await mem.init()

            sids = item["haystack_session_ids"]
            for idx, session in enumerate(item["haystack_sessions"]):
                date_prefix = ""
                if item.get("haystack_dates") and idx < len(item["haystack_dates"]):
                    date_prefix = f"[{item['haystack_dates'][idx]}] "

                turns = [f"{t['role']}: {t['content']}" for t in session]

                if chunk_mode == "whole":
                    text = date_prefix + "\n".join(turns)
                    if len(text) > trunc_limit:
                        text = text[:trunc_limit]
                    await mem.add(text, source_type="conv", source_ref=str(sids[idx]))

                elif chunk_mode in ("pairs", "pairs+summary"):
                    # Turn pairs
                    for j in range(0, len(turns), 2):
                        pair = turns[j : j + 2]
                        chunk = date_prefix + "\n".join(pair)
                        if len(chunk) > trunc_limit:
                            chunk = chunk[:trunc_limit]
                        await mem.add(
                            chunk, source_type="conv", source_ref=str(sids[idx])
                        )

                    if chunk_mode == "pairs+summary":
                        full = date_prefix + "\n".join(turns)
                        if len(full) > trunc_limit:
                            full = full[:trunc_limit]
                        await mem.add(
                            full,
                            source_type="conv",
                            source_ref=str(sids[idx]),
                            importance=summary_importance,
                        )

            results = await mem.search(item["question"], limit=top_k)

            # Session recall
            evidence = set(item["answer_session_ids"])
            retrieved_text = " ".join(r.fact.content for r in results).lower()
            sess_map = {}
            for idx2, sess2 in enumerate(item["haystack_sessions"]):
                key = sess2[0]["content"][:80].lower() if sess2 else ""
                sess_map[sids[idx2]] = key

            hit_count = 0
            for esid in evidence:
                key = sess_map.get(esid, "")
                if key and key in retrieved_text:
                    hit_count += 1
            sess_recall = hit_count / len(evidence) if evidence else 0
            total_recall += sess_recall

            await mem.close()

    return total_recall / len(data)


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function."""
    # Search space
    trunc_limit = trial.suggest_int("trunc_limit", 1000, 5000, step=500)
    top_k = trial.suggest_int("top_k", 5, 30, step=5)
    chunk_mode = trial.suggest_categorical(
        "chunk_mode", ["whole", "pairs", "pairs+summary"]
    )
    summary_importance = trial.suggest_float("summary_importance", 0.3, 1.0, step=0.1)

    data = trial.study.user_attrs["data"]
    recall = asyncio.run(
        eval_with_params(data, trunc_limit, top_k, chunk_mode, summary_importance)
    )
    return recall


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=60)
    args = parser.parse_args()

    print("Loading data...")
    data = load_sample()
    print(f"  {len(data)} questions loaded")

    study = optuna.create_study(direction="maximize", study_name="mmem-longmemeval")
    study.set_user_attr("data", data)

    # Seed with known good configs
    study.enqueue_trial(
        {
            "trunc_limit": 2000,
            "top_k": 10,
            "chunk_mode": "whole",
            "summary_importance": 0.5,
        }
    )  # baseline
    study.enqueue_trial(
        {
            "trunc_limit": 1500,
            "top_k": 10,
            "chunk_mode": "pairs",
            "summary_importance": 0.5,
        }
    )  # exp2
    study.enqueue_trial(
        {
            "trunc_limit": 1500,
            "top_k": 10,
            "chunk_mode": "pairs+summary",
            "summary_importance": 0.7,
        }
    )  # exp3
    study.enqueue_trial(
        {
            "trunc_limit": 1500,
            "top_k": 20,
            "chunk_mode": "pairs+summary",
            "summary_importance": 0.7,
        }
    )  # exp4

    print(f"\nRunning {args.n_trials} trials...\n")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print(f"\n{'=' * 60}")
    print(f"  OPTUNA SWEEP RESULTS ({args.n_trials} trials)")
    print(f"{'=' * 60}")
    print(f"  Best session recall: {study.best_value:.1%}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print(f"\n  Top 5 trials:")
    for t in sorted(study.trials, key=lambda x: x.value or 0, reverse=True)[:5]:
        print(f"    recall={t.value:.1%}  params={t.params}")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    results = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
            if t.value is not None
        ],
    }
    out = RESULTS_DIR / "optuna_sweep.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
