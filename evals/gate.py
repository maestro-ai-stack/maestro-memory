#!/usr/bin/env python3
"""Retrieval gate: replay real labelled queries and score ranking quality.

This is the arbiter for the mnerve -> mmem consolidation. Unlike
``run_retrieval_eval.py``, which seeds synthetic scenarios, this replays the
**real** query/feedback log exported from the mnerve Postgres store: 155
queries, 126 carrying agent-supplied relevance labels.

Two modes:

  --recorded          Score the ranking as it was actually served. Reproduces
                      the pre-migration baseline and proves the harness is
                      correct before it is allowed to judge anything else.

  --engine mmem       Run each query through maestro-memory and score the
                      result, comparing against the baseline.

Matching is deliberately prefix-based. mnerve truncates ``kg_nodes.name`` to
100 characters while the full text lives in ``properties.content``; a migrated
store holds the full text. Normalising both to a lowercase, whitespace-collapsed
100-character prefix makes the gold set portable across engines.

Usage:
    python evals/gate.py --recorded
    python evals/gate.py --engine mmem --baseline evals/results/gate_baseline.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_GOLD = Path.home() / ".maestro/backups/nerve-export-20260818/gold_queries.jsonl"
KEY_LEN = 100
_WS = re.compile(r"\s+")


# ── Key normalisation ─────────────────────────────────────────


def norm_key(text: str) -> str:
    """Collapse a fact to a comparable key.

    mnerve stores a 100-char truncation in ``name`` and the full string in
    ``properties.content``; both must reduce to the same key.
    """
    if not text:
        return ""
    return _WS.sub(" ", text).strip().lower()[:KEY_LEN]


def ref_key(ref: str | None) -> str:
    """A gold ref is 'node_type\\tname'; score on the name alone.

    Types differ between engines (mnerve node_type vs mmem fact_type), so
    including the type would make the gold set non-portable.
    """
    if not ref:
        return ""
    _, _, name = ref.partition("\t")
    return norm_key(name or ref)


# ── Metrics ───────────────────────────────────────────────────


@dataclass
class GateReport:
    n_queries: int = 0
    n_labelled: int = 0
    n_with_hit: int = 0
    p_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    mrr: float = 0.0
    misses: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "n_labelled": self.n_labelled,
            "n_with_hit": self.n_with_hit,
            "p_at_1": round(self.p_at_1, 4),
            "hit_at_3": round(self.hit_at_3, 4),
            "hit_at_5": round(self.hit_at_5, 4),
            "mrr": round(self.mrr, 4),
        }

    def render(self, title: str) -> str:
        d = self.as_dict()
        return (
            f"\n{title}\n"
            f"  queries        {d['n_queries']} ({d['n_labelled']} labelled)\n"
            f"  answer present {d['n_with_hit']}/{d['n_labelled']}"
            f" ({d['n_with_hit'] / max(d['n_labelled'], 1):.1%})\n"
            f"  P@1            {d['p_at_1']:.4f}\n"
            f"  Hit@3          {d['hit_at_3']:.4f}\n"
            f"  Hit@5          {d['hit_at_5']:.4f}\n"
            f"  MRR            {d['mrr']:.4f}\n"
        )


def score(cases: list[tuple[str, set[str], list[str]]]) -> GateReport:
    """Score (query_text, relevant_keys, ranked_keys) triples.

    Rank of the first relevant result drives every metric, matching the
    baseline computed from the original serving_log.
    """
    rep = GateReport(n_queries=len(cases))
    best_ranks: list[int | None] = []

    for query, relevant, ranked in cases:
        if not relevant:
            continue
        rep.n_labelled += 1
        best: int | None = None
        for i, key in enumerate(ranked):
            if key in relevant:
                best = i
                break
        best_ranks.append(best)
        if best is None:
            rep.misses.append({"query": query[:90], "n_relevant": len(relevant)})
        else:
            rep.n_with_hit += 1

    n = max(rep.n_labelled, 1)
    rep.p_at_1 = sum(1 for r in best_ranks if r == 0) / n
    rep.hit_at_3 = sum(1 for r in best_ranks if r is not None and r < 3) / n
    rep.hit_at_5 = sum(1 for r in best_ranks if r is not None and r < 5) / n
    rep.mrr = sum(1.0 / (r + 1) for r in best_ranks if r is not None) / n
    return rep


# ── Gold set ──────────────────────────────────────────────────


def load_gold(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"gold set not found: {path}\nRun migrate/export_from_nerve.sh first.")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def recorded_cases(gold: list[dict]) -> list[tuple[str, set[str], list[str]]]:
    """Rebuild the ranking exactly as it was served."""
    cases = []
    for row in gold:
        relevant = {ref_key(r) for r in (row.get("selected_refs") or [])}
        relevant.discard("")
        ranked = [
            ref_key(c.get("ref"))
            for c in sorted(row.get("candidates") or [], key=lambda c: c.get("rank", 999))
        ]
        cases.append((row.get("query_text") or "", relevant, ranked))
    return cases


# ── Engine adapters ───────────────────────────────────────────


async def mmem_cases(
    gold: list[dict], limit: int, db: Path | None = None
) -> list[tuple[str, set[str], list[str]]]:
    """Replay the gold set through maestro-memory.

    ``db`` should point at a scratch copy of the store: ``Memory.search`` writes
    a row to ``serving_logs`` per query, so running the gate against the live
    database would inject 155 synthetic rows into the training data on every run.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from maestro_memory import Memory  # noqa: PLC0415

    mem = Memory(path=db) if db else Memory()
    await mem.init()  # opens the store and builds the ANN index; search() fails without it
    cases = []
    for row in gold:
        query = row.get("query_text") or ""
        relevant = {ref_key(r) for r in (row.get("selected_refs") or [])}
        relevant.discard("")
        if not query or not relevant:
            cases.append((query, relevant, []))
            continue
        results = await mem.search(query, limit=limit)
        ranked = [norm_key(getattr(r.fact, "content", "") or "") for r in results]
        cases.append((query, relevant, ranked))
    return cases


# ── Entry point ───────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--recorded", action="store_true", help="score the ranking as served (baseline)")
    ap.add_argument("--engine", choices=["mmem"], help="score a live engine")
    ap.add_argument("--db", type=Path, help="engine DB path; use a scratch copy — search() writes serving_logs")
    ap.add_argument("--limit", type=int, default=15, help="results per query (gold averages 14.9)")
    ap.add_argument("--baseline", type=Path, help="compare against a saved report; non-zero exit on regression")
    ap.add_argument("--save", type=Path, help="write the report as JSON")
    ap.add_argument("--tolerance", type=float, default=0.0, help="allowed MRR drop before failing")
    args = ap.parse_args()

    if not args.recorded and not args.engine:
        ap.error("choose --recorded or --engine")

    gold = load_gold(args.gold)

    if args.recorded:
        rep = score(recorded_cases(gold))
        title = f"RECORDED (as served) — {args.gold.name}"
    else:
        rep = score(asyncio.run(mmem_cases(gold, args.limit, args.db)))
        title = f"ENGINE=mmem ({args.db or 'default store'}) — {args.gold.name}"

    print(rep.render(title))
    if rep.misses:
        print(f"  {len(rep.misses)} labelled queries with no relevant result in range")
        for m in rep.misses[:5]:
            print(f"    - {m['query']}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(rep.as_dict(), indent=2) + "\n")
        print(f"  saved -> {args.save}")

    if args.baseline:
        if not args.baseline.exists():
            sys.exit(f"baseline not found: {args.baseline}")
        base = json.loads(args.baseline.read_text())
        drop = base["mrr"] - rep.mrr
        print(f"\n  baseline MRR {base['mrr']:.4f} -> {rep.mrr:.4f} (delta {-drop:+.4f})")
        if drop > args.tolerance:
            print("  GATE FAIL: ranking regressed")
            return 1
        print("  GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
