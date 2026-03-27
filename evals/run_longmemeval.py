#!/usr/bin/env python3
"""LongMemEval benchmark adapter for maestro-memory.

Evaluates mmem's retrieval quality on the LongMemEval benchmark (ICLR 2025).
Protocol:
  1. For each question, ingest all chat sessions into a fresh mmem DB
  2. Search mmem with the question
  3. Check if retrieved facts overlap with evidence sessions (retrieval recall)
  4. Optionally generate answer via LLM and evaluate with GPT-4o

Usage:
  # Retrieval-only eval (no LLM needed, fast)
  python evals/run_longmemeval.py --mode retrieval --limit 50

  # Full QA eval (needs OPENAI_API_KEY for answer generation + judging)
  python evals/run_longmemeval.py --mode qa --limit 50

  # Run on oracle dataset (evidence sessions only, easier)
  python evals/run_longmemeval.py --data oracle --limit 50
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from maestro_memory import Memory


# ── Paths ──────────────────────────────────────────────────────────
LONGMEMEVAL_DIR = Path.home() / "maestro/projects/LongMemEval/data"
RESULTS_DIR = Path(__file__).parent / "results"


def load_dataset(variant: str = "oracle") -> list[dict]:
    """Load LongMemEval dataset."""
    files = {
        "oracle": "longmemeval_oracle.json",
        "s": "longmemeval_s_cleaned.json",
    }
    path = LONGMEMEVAL_DIR / files[variant]
    if not path.exists():
        print(f"ERROR: {path} not found. Run:")
        print(f"  cd {LONGMEMEVAL_DIR}")
        print(f"  curl -sL https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/{files[variant]} -o {files[variant]}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def sessions_to_text(sessions: list[list[dict]], dates: list[str] | None = None) -> list[str]:
    """Convert chat sessions to text chunks for ingestion."""
    chunks = []
    for i, session in enumerate(sessions):
        date_prefix = f"[{dates[i]}] " if dates and i < len(dates) else ""
        lines = []
        for turn in session:
            role = turn["role"]
            content = turn["content"]
            lines.append(f"{role}: {content}")
        text = date_prefix + "\n".join(lines)
        # Truncate very long sessions to avoid overwhelming single facts
        if len(text) > 2000:
            text = text[:2000] + "..."
        chunks.append(text)
    return chunks


async def eval_retrieval_single(item: dict, top_k: int = 10) -> dict:
    """Evaluate retrieval for a single question."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "eval.db"
        mem = Memory(path=db_path)
        await mem.init()

        # Ingest all sessions
        sessions_text = sessions_to_text(
            item["haystack_sessions"],
            item.get("haystack_dates"),
        )
        session_id_map = {}  # fact_content -> session_id
        for idx, (text, sid) in enumerate(
            zip(sessions_text, item["haystack_session_ids"])
        ):
            await mem.add(text, source_type="conversation", source_ref=str(sid))
            session_id_map[text[:200]] = sid

        # Search
        t0 = time.perf_counter()
        results = await mem.search(item["question"], limit=top_k)
        search_time = time.perf_counter() - t0

        # Check retrieval recall: do retrieved facts come from evidence sessions?
        evidence_sids = set(item["answer_session_ids"])
        retrieved_contents = [r.fact.content for r in results]

        # Map retrieved facts back to session IDs
        retrieved_sids = set()
        for fact_content in retrieved_contents:
            for text_prefix, sid in session_id_map.items():
                if text_prefix[:100] in fact_content or fact_content[:100] in text_prefix:
                    retrieved_sids.add(sid)
                    break

        # Session-level recall: how many evidence sessions were retrieved?
        if evidence_sids:
            hits = evidence_sids & retrieved_sids
            session_recall = len(hits) / len(evidence_sids)
        else:
            session_recall = 0.0

        # Content-level: check if answer keywords appear in retrieved text
        answer = item["answer"].lower()
        all_retrieved = " ".join(retrieved_contents).lower()
        # Simple keyword overlap
        answer_words = set(answer.split())
        retrieved_words = set(all_retrieved.split())
        if answer_words:
            keyword_overlap = len(answer_words & retrieved_words) / len(answer_words)
        else:
            keyword_overlap = 0.0

        stats = await mem.status()
        await mem.close()

    return {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "question": item["question"],
        "answer": item["answer"],
        "session_recall": session_recall,
        "keyword_overlap": keyword_overlap,
        "search_time_ms": search_time * 1000,
        "num_facts": stats["facts"],
        "num_sessions_ingested": len(sessions_text),
        "num_evidence_sessions": len(evidence_sids),
        "num_retrieved": len(results),
        "retrieved_preview": [r.fact.content[:100] for r in results[:3]],
    }


async def run_retrieval_eval(
    data: list[dict],
    limit: int | None = None,
    top_k: int = 10,
) -> list[dict]:
    """Run retrieval eval on dataset."""
    if limit:
        data = data[:limit]

    results = []
    for i, item in enumerate(data):
        result = await eval_retrieval_single(item, top_k=top_k)
        results.append(result)

        status = "HIT" if result["session_recall"] > 0 else "MISS"
        print(
            f"  [{i+1:3d}/{len(data)}] [{status}] "
            f"type={result['question_type']:<28s} "
            f"sess_recall={result['session_recall']:.0%} "
            f"kw_overlap={result['keyword_overlap']:.0%} "
            f"time={result['search_time_ms']:.0f}ms "
            f"facts={result['num_facts']}"
        )

    return results


def print_summary(results: list[dict], variant: str) -> None:
    """Print benchmark summary."""
    from collections import defaultdict

    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)

    print(f"\n{'=' * 72}")
    print(f"  LONGMEMEVAL ({variant.upper()}) — mmem RETRIEVAL BENCHMARK")
    print(f"{'=' * 72}")
    print(
        f"  {'Type':<30s} {'Count':>6} {'Sess Recall':>12} {'KW Overlap':>11} {'Avg ms':>8}"
    )
    print(f"  {'-' * 30} {'-' * 6} {'-' * 12} {'-' * 11} {'-' * 8}")

    total_recall = 0
    total_kw = 0
    total_time = 0
    n = len(results)

    for qtype in sorted(by_type.keys()):
        items = by_type[qtype]
        avg_recall = sum(r["session_recall"] for r in items) / len(items)
        avg_kw = sum(r["keyword_overlap"] for r in items) / len(items)
        avg_time = sum(r["search_time_ms"] for r in items) / len(items)
        total_recall += sum(r["session_recall"] for r in items)
        total_kw += sum(r["keyword_overlap"] for r in items)
        total_time += sum(r["search_time_ms"] for r in items)
        print(
            f"  {qtype:<30s} {len(items):>6} {avg_recall:>11.1%} {avg_kw:>10.1%} {avg_time:>7.0f}"
        )

    print(f"  {'-' * 30} {'-' * 6} {'-' * 12} {'-' * 11} {'-' * 8}")
    print(
        f"  {'OVERALL':<30s} {n:>6} "
        f"{total_recall / n:>11.1%} {total_kw / n:>10.1%} {total_time / n:>7.0f}"
    )
    print(f"{'=' * 72}")


async def main():
    parser = argparse.ArgumentParser(description="LongMemEval benchmark for mmem")
    parser.add_argument(
        "--data",
        choices=["oracle", "s"],
        default="oracle",
        help="Dataset variant: oracle (evidence only) or s (~40 sessions)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max questions to evaluate (default: all 500)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of facts to retrieve per query",
    )
    parser.add_argument(
        "--mode",
        choices=["retrieval"],
        default="retrieval",
        help="Evaluation mode",
    )
    args = parser.parse_args()

    print(f"Loading LongMemEval ({args.data})...")
    data = load_dataset(args.data)
    print(f"  {len(data)} questions loaded")

    if args.limit:
        print(f"  Limited to first {args.limit} questions")

    print(f"\nRunning retrieval evaluation (top-{args.top_k})...\n")
    results = await run_retrieval_eval(data, limit=args.limit, top_k=args.top_k)

    print_summary(results, args.data)

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"longmemeval_{args.data}_retrieval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
