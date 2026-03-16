#!/usr/bin/env python3
"""A/B eval runner for maestro-memory skill.

For each scenario:
  1. Seeds mmem with test data (fresh DB per scenario)
  2. Runs Claude with skill (mmem available) and without
  3. Grades outputs against assertions
  4. Outputs comparison report
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCENARIOS_PATH = Path(__file__).parent / "scenarios.json"
SKILL_PATH = Path(__file__).parent.parent / "skill" / "SKILL.md"


# ── Data preparation ──────────────────────────────────────────────

def seed_scenario(scenario: dict, db_path: Path) -> None:
    """Run setup commands to seed test data into the given DB."""
    env = {**os.environ, "MAESTRO_MEMORY_DB": str(db_path)}
    for cmd in scenario["setup"]:
        subprocess.run(cmd, shell=True, env=env, capture_output=True)


# ── Claude invocation ────────────────────────────────────────────

def run_claude(query: str, *, with_skill: bool, db_path: Path, timeout: int = 120) -> str:
    """Call claude -p and return output text."""
    if with_skill:
        prompt = (
            f"You have access to maestro-memory via the `mmem` CLI tool. "
            f"The database is at {db_path}. "
            f"Use `mmem search` to find relevant information before answering.\n\n"
            f"Question: {query}"
        )
    else:
        prompt = f"Answer this question based on your knowledge. You have NO memory system available.\n\nQuestion: {query}"

    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    env["MAESTRO_MEMORY_DB"] = str(db_path)

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except FileNotFoundError:
        return "[ERROR: claude CLI not found]"


# ── Grading ──────────────────────────────────────────────────────

def grade_output(output: str, assertions: list[str]) -> list[dict]:
    """Simple keyword-matching grader (production should use LLM-as-judge)."""
    results = []
    output_lower = output.lower()
    for assertion in assertions:
        # Extract keywords (quoted or key phrases)
        keywords = extract_keywords(assertion)
        passed = any(kw.lower() in output_lower for kw in keywords)
        results.append({"assertion": assertion, "passed": passed, "keywords": keywords})
    return results


def extract_keywords(assertion: str) -> list[str]:
    """Extract check keywords from an assertion."""
    keywords = []
    # Extract quoted content
    import re
    quoted = re.findall(r'[\$]?[\d,]+[KkMm]?', assertion)
    keywords.extend(quoted)
    # Extract capitalized words or proper nouns
    words = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', assertion)
    keywords.extend(words)
    # Extract key English phrases
    for phrase in ["shellfish", "dizziness", "headache", "losartan", "lisinopril",
                   "sigmoid", "jenkins", "cache warming", "batch processing",
                   "Neo4j", "SQLite", "ACT-R", "Ebbinghaus", "Datadog",
                   "Park Hyatt", "Hotel Sunrise", "Dotonbori", "Tsukiji",
                   "token bucket", "rate limiter", "Kubernetes", "k8s"]:
        if phrase.lower() in assertion.lower():
            keywords.append(phrase)
    # Fallback: use the longest word in the assertion
    if not keywords:
        words = assertion.split()
        keywords = [max(words, key=len)] if words else [""]
    return keywords


# ── Main flow ────────────────────────────────────────────────────

def run_scenario(scenario: dict, results_dir: Path) -> dict:
    """Run A/B comparison for a single scenario."""
    sid = scenario["id"]
    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario['name']} ({sid})")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "eval.db"

        # Seed data
        print(f"  Seeding {len(scenario['setup'])} facts...")
        seed_scenario(scenario, db_path)

        # with skill
        print(f"  Running WITH skill...")
        with_output = run_claude(scenario["query"], with_skill=True, db_path=db_path)

        # without skill
        print(f"  Running WITHOUT skill...")
        without_output = run_claude(scenario["query"], with_skill=False, db_path=db_path)

    # Grade outputs
    with_grades = grade_output(with_output, scenario["assertions"])
    without_grades = grade_output(without_output, scenario["assertions"])

    with_pass = sum(1 for g in with_grades if g["passed"])
    without_pass = sum(1 for g in without_grades if g["passed"])
    total = len(scenario["assertions"])

    result = {
        "id": sid, "name": scenario["name"],
        "with_skill": {"output": with_output, "grades": with_grades, "pass_rate": with_pass / total},
        "without_skill": {"output": without_output, "grades": without_grades, "pass_rate": without_pass / total},
        "delta": (with_pass - without_pass) / total,
    }

    # Save result
    out_path = results_dir / f"{sid}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # Print summary
    print(f"  WITH skill:    {with_pass}/{total} ({result['with_skill']['pass_rate']:.0%})")
    print(f"  WITHOUT skill: {without_pass}/{total} ({result['without_skill']['pass_rate']:.0%})")
    print(f"  Delta:         {result['delta']:+.0%}")

    return result


def main():
    scenarios = json.loads(SCENARIOS_PATH.read_text())

    # Filter by scenario ID if args provided
    if len(sys.argv) > 1:
        ids = sys.argv[1:]
        scenarios = [s for s in scenarios if s["id"] in ids]

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    all_results = []
    for scenario in scenarios:
        result = run_scenario(scenario, results_dir)
        all_results.append(result)

    # Summary report
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Scenario':<25} {'With':>8} {'Without':>8} {'Delta':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")

    total_with = total_without = 0
    for r in all_results:
        w = r["with_skill"]["pass_rate"]
        wo = r["without_skill"]["pass_rate"]
        total_with += w
        total_without += wo
        print(f"  {r['name']:<25} {w:>7.0%} {wo:>7.0%} {r['delta']:>+7.0%}")

    n = len(all_results)
    if n:
        avg_w = total_with / n
        avg_wo = total_without / n
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
        print(f"  {'AVERAGE':<25} {avg_w:>7.0%} {avg_wo:>7.0%} {avg_w - avg_wo:>+7.0%}")

    # Save summary
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\n  Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
