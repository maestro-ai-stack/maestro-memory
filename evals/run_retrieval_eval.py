#!/usr/bin/env python3
"""Retrieval benchmark for maestro-memory.

Directly evaluate mmem search retrieval quality without Claude LLM.
Each scenario: seed data -> search -> check if top-k contains expected facts.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Load modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from maestro_memory import Memory


# ── Eval scenarios ────────────────────────────────────────────────

SCENARIOS = [
    {
        "name": "Multi-turn Extraction",
        "facts": [
            "Project Alpha launched on March 1, budget $200K, lead: Sarah Chen",
            "Alpha milestone 1 completed March 5, on budget",
            "Alpha hit blocker March 8: vendor API rate limit, switched to batch processing",
            "Alpha milestone 2 delayed to March 15 due to API issue",
            "Alpha back on track March 12, new ETA March 18 for final delivery",
        ],
        "queries": [
            ("Alpha budget", ["$200K"]),
            ("Alpha blocker", ["API rate limit", "batch processing"]),
            ("Alpha ETA", ["March 18"]),
            ("Sarah Chen", ["Sarah Chen", "Alpha"]),
        ],
    },
    {
        "name": "CRM Extraction",
        "facts": [
            "Alice Wang from Acme Corp, deal worth $50K, stage: negotiation",
            "Bob Lee from Beta Inc, deal $20K, stage: closed-won, signed March 10",
            "Carol Zhang from Gamma Ltd, deal $80K, stage: discovery, meeting March 20",
            "Alice Wang update: Acme deal increased to $75K after adding enterprise tier",
            "Bob Lee churned from Beta Inc, contract cancelled March 15",
        ],
        "queries": [
            ("Acme Corp", ["Acme", "Alice Wang"]),
            ("closed deals", ["closed-won", "Beta"]),
            ("Gamma Ltd", ["Gamma", "Carol", "$80K"]),
            ("enterprise tier", ["$75K", "enterprise"]),
        ],
    },
    {
        "name": "Calendar Scheduling",
        "facts": [
            "Team standup every Monday Wednesday Friday 10am, room 301",
            "Dentist appointment March 20 at 2pm, Dr. Li clinic",
            "Client demo March 20 at 3pm with Acme Corp, need projector",
            "Lunch with mentor March 21 at 12pm, restaurant TBD",
            "Client demo moved from March 20 to March 22 at 3pm per client request",
        ],
        "queries": [
            ("March 20", ["March 20", "dentist"]),
            ("client demo", ["demo", "March 22"]),
            ("standup schedule", ["standup", "Monday"]),
        ],
    },
    {
        "name": "Travel Planning",
        "facts": [
            "Travel preferences: window seat, vegetarian meals, budget max $3000",
            "Allergic to shellfish, must avoid seafood restaurants",
            "Japan trip: Tokyo March 15-20, Hotel Sunrise Shinjuku",
            "Japan trip: Osaka March 20-23, Namba Grand hotel",
            "Must visit Tsukiji outer market Tokyo, Dotonbori Osaka",
            "Tokyo hotel changed to Park Hyatt Shinjuku, upgrade from company",
        ],
        "queries": [
            ("food restrictions", ["shellfish", "vegetarian"]),
            ("Tokyo hotel", ["Park Hyatt", "Tokyo"]),
            ("Osaka", ["Osaka", "Namba", "Dotonbori"]),
            ("budget", ["$3000"]),
        ],
    },
    {
        "name": "Patient Records",
        "facts": [
            "Patient John Doe, 45M, persistent headache 3 days, no fever",
            "John Doe vitals: BP 140/90, HR 78. History: hypertension, lisinopril 10mg daily",
            "John Doe: tension headache. Plan: ibuprofen 400mg TID x 5 days",
            "Follow-up March 12: headache resolved. New: intermittent dizziness 2 days",
            "March 12: possible medication side effect. Switched lisinopril to losartan 50mg",
        ],
        "queries": [
            ("John Doe medication", ["losartan", "lisinopril"]),
            ("dizziness", ["dizziness", "side effect"]),
            ("blood pressure", ["BP", "140/90", "hypertension"]),
        ],
    },
    {
        "name": "HR Candidate",
        "facts": [
            "Emily Chen, 8 years Python/Go, applied Senior Backend Engineer. Stripe background",
            "Emily Chen phone screen: strong system design. Weakness: limited Kubernetes",
            "Emily Chen onsite: coding excellent, graph problem O(n). Mike recommends hire",
            "Emily Chen system design: rate limiter with token bucket. Gap: monitoring",
            "Emily Chen: extend offer $180K + equity. Competing offer from Datadog",
        ],
        "queries": [
            ("Emily Chen strengths", ["system design", "coding"]),
            ("Emily Chen weaknesses", ["Kubernetes"]),
            ("Emily offer", ["$180K", "Datadog"]),
            ("Emily interview", ["Mike", "recommends"]),
        ],
    },
    {
        "name": "Paper Management",
        "facts": [
            "Attention Is All You Need (Vaswani 2017): transformer, self-attention replacing RNNs",
            "BERT (Devlin 2019): bidirectional pre-training, masked language modeling",
            "GPT-3 (Brown 2020): 175B params, few-shot learning at scale",
            "Graphiti (Zep 2025): temporal knowledge graph, bi-temporal, Neo4j",
            "Design decision: chose SQLite over Neo4j for zero-infrastructure constraint",
        ],
        "queries": [
            ("transformer", ["transformer", "Vaswani"]),
            ("knowledge graph memory", ["Graphiti", "temporal"]),
            ("SQLite decision", ["SQLite", "Neo4j", "zero"]),
        ],
    },
    {
        "name": "Research Tracking",
        "facts": [
            "Research question: Does ACT-R improve retrieval over Ebbinghaus decay?",
            "Finding: Ebbinghaus curve too aggressive, 100-day facts invisible",
            "Finding: ACT-R base-level activation balances recency and frequency",
            "Dead end: frequency-only boosting buries important rarely-accessed facts",
            "Conclusion: ACT-R with importance weighting, score = importance * sigmoid(A)",
        ],
        "queries": [
            ("ACT-R", ["ACT-R", "activation"]),
            ("Ebbinghaus problem", ["Ebbinghaus", "aggressive"]),
            ("dead end research", ["frequency", "dead end"]),
            ("final conclusion", ["sigmoid", "importance"]),
        ],
    },
    {
        "name": "Work Project Chat",
        "facts": [
            "deploy blocked bc jenkins down, mike looking into it, eta 2hrs",
            "update: jenkins back, deploy succeeded to staging, monitoring 30min",
            "staging looks good, health checks green, pushing to prod",
            "prod deploy done. elevated p99 latency /api/search 800ms vs 200ms",
            "root cause: search index not warmed. cache warming job, eta 1hr",
        ],
        "queries": [
            ("jenkins", ["jenkins", "blocked"]),
            ("deploy status", ["deploy", "prod"]),
            ("latency issue", ["p99", "latency", "800ms"]),
            ("cache warming", ["cache", "warming", "1hr"]),
        ],
    },
]


# ── Eval engine ────────────────────────────────────────────────────

async def eval_scenario(scenario: dict) -> dict:
    """Evaluate a single scenario."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "eval.db"
        mem = Memory(path=db_path)
        await mem.init()

        # Seed data
        for fact in scenario["facts"]:
            await mem.add(fact, source_type="eval")

        # Retrieval evaluation
        query_results = []
        for query, expected_keywords in scenario["queries"]:
            results = await mem.search(query, limit=5)
            # Check if top-5 results contain expected keywords
            all_content = " ".join(r.fact.content for r in results).lower()
            hits = [kw for kw in expected_keywords if kw.lower() in all_content]
            query_results.append({
                "query": query,
                "expected": expected_keywords,
                "hits": hits,
                "miss": [kw for kw in expected_keywords if kw not in hits],
                "recall": len(hits) / len(expected_keywords) if expected_keywords else 0,
                "top_results": [r.fact.content[:80] for r in results[:3]],
            })

        await mem.close()

    avg_recall = sum(q["recall"] for q in query_results) / len(query_results)
    return {
        "name": scenario["name"],
        "queries": query_results,
        "avg_recall": avg_recall,
        "total_queries": len(query_results),
    }


async def main():
    # Filter by scenario name if args provided
    scenarios = SCENARIOS
    if len(sys.argv) > 1:
        names = {s.lower() for s in sys.argv[1:]}
        scenarios = [s for s in SCENARIOS if any(n in s["name"].lower() for n in names)]

    print(f"Running {len(scenarios)} scenarios...\n")

    results = []
    for scenario in scenarios:
        result = await eval_scenario(scenario)
        results.append(result)

        # Print details
        status = "PASS" if result["avg_recall"] >= 0.8 else "FAIL"
        print(f"[{status}] {result['name']} — recall: {result['avg_recall']:.0%}")
        for q in result["queries"]:
            qstatus = "ok" if q["recall"] == 1.0 else "MISS"
            print(f"  [{qstatus}] '{q['query']}' → {q['hits']} | miss: {q['miss']}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  RETRIEVAL BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Scenario':<25} {'Recall':>8} {'Queries':>8} {'Status':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")

    total_recall = 0
    passed = 0
    for r in results:
        status = "PASS" if r["avg_recall"] >= 0.8 else "FAIL"
        if r["avg_recall"] >= 0.8:
            passed += 1
        total_recall += r["avg_recall"]
        print(f"  {r['name']:<25} {r['avg_recall']:>7.0%} {r['total_queries']:>8} {status:>8}")

    n = len(results)
    avg = total_recall / n if n else 0
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'OVERALL':<25} {avg:>7.0%} {sum(r['total_queries'] for r in results):>8} {passed}/{n}")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "retrieval_benchmark.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    print(f"\n  Saved to {results_dir}/retrieval_benchmark.json")


if __name__ == "__main__":
    asyncio.run(main())
