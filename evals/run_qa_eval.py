#!/usr/bin/env python3
"""LongMemEval QA evaluation for maestro-memory.

Full pipeline: ingest sessions → retrieve → LLM generate answer → GPT-4o judge.
This produces scores directly comparable to Mem0/Zep/Hindsight.

Usage:
  python evals/run_qa_eval.py --limit 100 --tag baseline
  python evals/run_qa_eval.py --limit 100 --tag bgem3 --compare baseline
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

# Load API keys from known .env files
def load_env_keys():
    """Load API keys from project .env files."""
    import os
    env_files = [
        Path.home() / "maestro/projects/2026-1-ra-suite/.env.local",
        Path.home() / "maestro/projects/2026-1-ra-suite/gateway-py/.env",
        Path.home() / "maestro/projects/2025-11-ra-experiment/.env.local",
    ]
    for env_file in env_files:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") and val:
                        os.environ.setdefault(key, val)

load_env_keys()

import os
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from maestro_memory import Memory

LONGMEMEVAL_DIR = Path.home() / "maestro/projects/LongMemEval/data"
RESULTS_DIR = Path(__file__).parent / "results"


# ── Answer generation (Claude) ──────────────────────────────────

def generate_answer_claude(question: str, context: str) -> str:
    """Generate answer using Claude given retrieved context."""
    import anthropic
    client = anthropic.Anthropic()

    prompt = f"""Based on the following conversation history excerpts, answer the user's question.
If the information is not available in the excerpts, say "I don't have enough information to answer that."

Conversation history excerpts:
{context}

Question: {question}

Answer concisely and directly."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    block = response.content[0]
    return block.text if hasattr(block, "text") else str(block)


# ── GPT-4o judge (LongMemEval standard) ─────────────────────────

def get_judge_prompt(task: str, question: str, answer: str, response: str) -> str:
    """Get judge prompt matching LongMemEval's evaluate_qa.py exactly."""
    abstention = question.endswith("_abs") if False else False  # handled by question_id

    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
    elif task == "temporal-reasoning":
        template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
    elif task == "knowledge-update":
        template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
    elif task == "single-session-preference":
        template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
    else:
        template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."

    return template.format(question, answer, response)


def judge_answer(task: str, question: str, gold_answer: str, model_answer: str, question_id: str) -> bool:
    """Use GPT-4o to judge if the answer is correct."""
    from openai import OpenAI
    client = OpenAI()

    abstention = question_id.endswith("_abs")

    if abstention:
        prompt = f"I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {question}\n\nExplanation: {gold_answer}\n\nModel Response: {model_answer}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    else:
        prompt = get_judge_prompt(task, question, gold_answer, model_answer)

    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0,
    )
    verdict = response.choices[0].message.content.strip().lower()
    return verdict.startswith("yes")


# ── Main evaluation ─────────────────────────────────────────────

async def eval_single(item: dict, top_k: int = 10) -> dict:
    """Full QA eval for a single question."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "eval.db"
        mem = Memory(path=db_path)
        await mem.init()

        # Ingest all sessions
        sids = item["haystack_session_ids"]
        for idx, session in enumerate(item["haystack_sessions"]):
            date_prefix = ""
            if item.get("haystack_dates") and idx < len(item["haystack_dates"]):
                date_prefix = f"[{item['haystack_dates'][idx]}] "
            text = date_prefix + "\n".join(
                f"{t['role']}: {t['content']}" for t in session
            )
            if len(text) > 2000:
                text = text[:2000]
            await mem.add(text, source_type="conversation", source_ref=str(sids[idx]))

        # Retrieve
        t0 = time.perf_counter()
        results = await mem.search(item["question"], limit=top_k)
        retrieve_ms = (time.perf_counter() - t0) * 1000

        await mem.close()

    # Build context from retrieved facts
    context = "\n\n---\n\n".join(r.fact.content for r in results)

    # Truncate context to avoid token limits
    if len(context) > 12000:
        context = context[:12000] + "\n...[truncated]"

    # Generate answer
    t0 = time.perf_counter()
    try:
        model_answer = generate_answer_claude(item["question"], context)
    except Exception as e:
        model_answer = f"Error generating answer: {e}"
    generate_ms = (time.perf_counter() - t0) * 1000

    # Judge
    t0 = time.perf_counter()
    try:
        correct = judge_answer(
            item["question_type"],
            item["question"],
            item["answer"],
            model_answer,
            item["question_id"],
        )
    except Exception as e:
        print(f"    Judge error: {e}")
        correct = False
    judge_ms = (time.perf_counter() - t0) * 1000

    return {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "question": item["question"],
        "gold_answer": item["answer"],
        "model_answer": model_answer,
        "correct": correct,
        "retrieve_ms": retrieve_ms,
        "generate_ms": generate_ms,
        "judge_ms": judge_ms,
    }


async def run_eval(data: list[dict], limit: int | None, top_k: int, tag: str) -> list[dict]:
    """Run full QA evaluation."""
    if limit:
        data = data[:limit]

    results = []
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, item in enumerate(data):
        result = await eval_single(item, top_k=top_k)
        results.append(result)

        qtype = result["question_type"]
        by_type[qtype]["total"] += 1
        if result["correct"]:
            by_type[qtype]["correct"] += 1

        status = "✓" if result["correct"] else "✗"
        running_acc = sum(1 for r in results if r["correct"]) / len(results)
        print(
            f"  [{i+1:3d}/{len(data)}] {status} "
            f"type={qtype:<28s} "
            f"running_acc={running_acc:.1%} "
            f"ret={result['retrieve_ms']:.0f}ms "
            f"gen={result['generate_ms']:.0f}ms"
        )

    return results


def print_summary(results: list[dict], tag: str, compare_path: Path | None = None) -> None:
    """Print and optionally compare results."""
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_type[r["question_type"]]["total"] += 1
        if r["correct"]:
            by_type[r["question_type"]]["correct"] += 1

    # Load comparison if provided
    compare = None
    if compare_path and compare_path.exists():
        with open(compare_path) as f:
            compare_results = json.load(f)
        compare = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in compare_results:
            compare[r["question_type"]]["total"] += 1
            if r["correct"]:
                compare[r["question_type"]]["correct"] += 1

    print(f"\n{'=' * 78}")
    print(f"  LONGMEMEVAL QA ACCURACY — [{tag}]")
    print(f"{'=' * 78}")

    header = f"  {'Type':<30s} {'Count':>6} {'Accuracy':>9}"
    if compare:
        header += f" {'Baseline':>9} {'Delta':>8}"
    print(header)
    print(f"  {'-' * 30} {'-' * 6} {'-' * 9}" + (f" {'-' * 9} {'-' * 8}" if compare else ""))

    total_correct = sum(d["correct"] for d in by_type.values())
    total_n = sum(d["total"] for d in by_type.values())

    for qtype in sorted(by_type.keys()):
        d = by_type[qtype]
        acc = d["correct"] / d["total"] if d["total"] else 0
        line = f"  {qtype:<30s} {d['total']:>6} {acc:>8.1%}"
        if compare and qtype in compare:
            base_acc = compare[qtype]["correct"] / compare[qtype]["total"]
            delta = acc - base_acc
            line += f" {base_acc:>8.1%} {delta:>+7.1%}"
        print(line)

    print(f"  {'-' * 30} {'-' * 6} {'-' * 9}" + (f" {'-' * 9} {'-' * 8}" if compare else ""))
    overall = total_correct / total_n if total_n else 0
    line = f"  {'OVERALL':<30s} {total_n:>6} {overall:>8.1%}"
    if compare:
        base_total = sum(d["correct"] for d in compare.values())
        base_n = sum(d["total"] for d in compare.values())
        base_overall = base_total / base_n if base_n else 0
        delta = overall - base_overall
        line += f" {base_overall:>8.1%} {delta:>+7.1%}"
    print(line)
    print(f"{'=' * 78}")

    # Comparison with known benchmarks
    print(f"\n  Comparison with known systems:")
    print(f"  {'System':<20s} {'QA Accuracy':>12}")
    print(f"  {'-' * 20} {'-' * 12}")
    print(f"  {'mmem [' + tag + ']':<20s} {overall:>11.1%}")
    print(f"  {'Mem0':<20s} {'49.0%':>12}")
    print(f"  {'Zep':<20s} {'63.8%':>12}")
    print(f"  {'Hindsight':<20s} {'91.4%':>12}")


async def main():
    parser = argparse.ArgumentParser(description="LongMemEval QA eval for mmem")
    parser.add_argument("--limit", type=int, default=100, help="Max questions (default 100)")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval top-k")
    parser.add_argument("--tag", type=str, default="baseline", help="Run tag for comparison")
    parser.add_argument("--compare", type=str, default=None, help="Tag to compare against")
    parser.add_argument("--data", choices=["oracle", "s"], default="s", help="Dataset variant")
    args = parser.parse_args()

    # Load data
    files = {"oracle": "longmemeval_oracle.json", "s": "longmemeval_s_cleaned.json"}
    data_path = LONGMEMEVAL_DIR / files[args.data]
    print(f"Loading LongMemEval ({args.data})...")
    with open(data_path) as f:
        data = json.load(f)
    print(f"  {len(data)} questions, using first {args.limit}")

    # Run
    print(f"\nRunning QA evaluation [tag={args.tag}]...\n")
    results = await run_eval(data, limit=args.limit, top_k=args.top_k, tag=args.tag)

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"qa_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Compare
    compare_path = RESULTS_DIR / f"qa_{args.compare}.json" if args.compare else None
    print_summary(results, args.tag, compare_path)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
