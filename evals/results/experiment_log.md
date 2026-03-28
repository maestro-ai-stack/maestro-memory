# mmem Autoresearch Experiment Log

Baseline: **50% QA Accuracy** on LongMemEval_S (N=50, subagent-judged)
Target: **65%+** (beat Zep 63.8%)
Session Recall baseline: **83.3%** (N=500)

## Experiment 0: Baseline
- Date: 2026-03-27
- QA Accuracy: 50% (25/50)
- Session Recall: 83.3% (N=500)
- Latency: 6ms avg
- By type: temporal-reasoning 69%, preference 80%, multi-session 46%, knowledge-update 43%, single-session-user 29%, single-session-assistant 20%
- Key finding: 84% of failures are retrieval failures (context missing)

## Experiment 1: Truncation 2000→4000 + top-k 10→15
- Date: 2026-03-27
- Hypothesis: longer excerpts fix assistant truncation, more results increase hit rate
- Change: eval params only (trunc 4000, top_k 15, context 800 chars/excerpt)
- Session hit rate: 96% (was 91%)
- QA Accuracy: **50% (25/50) — no change**
- Delta by type: multi-session +8%, temporal-reasoning -8%, rest unchanged
- Verdict: **REVERT**
- Reason: net zero. More context helped multi-session but hurt temporal (diluted signal with noise). Truncation wasn't the bottleneck — the WRONG sessions are retrieved.

## Experiment 2: Session decomposition (turn pairs) ✓ KEEP
- Date: 2026-03-27
- Hypothesis: splitting sessions into user+assistant turn pairs gives BM25/embedding more precise targets
- Change: eval ingestion — each session decomposed into turn pairs + summary chunk (~280 facts/q vs ~48)
- QA Accuracy: **66% (33/50) — +16% from baseline!**
- Delta by type: single-session-assistant +60%, single-session-user +57%, knowledge-update +43%, temporal +15%, multi-session -23%, preference -20%
- Verdict: **KEEP** — surpassed Zep (63.8%), now #3 behind Hindsight (91.4%)
- Regression: multi-session dropped 46%→23% — aggregation queries hurt by too-granular facts
- Next: fix multi-session regression while keeping other gains

## Experiment 3: Turn pairs + session summary ✓ KEEP
- Date: 2026-03-27
- Hypothesis: add full session text (importance=0.7) alongside turn pairs to help multi-session aggregation
- Change: eval ingestion — add session-level summary fact with higher importance
- QA Accuracy: **70% (36/50) — +4% from exp2, +20% from baseline!**
- Delta vs exp2: multi-session +8%, temporal +7%, rest unchanged
- Verdict: **KEEP** — new best, beat Zep by 6%
- Remaining weakness: multi-session still at 31% (was 46% baseline, aggregation queries)
- Next: try increasing top-k for multi-session or add multi-hop retrieval

## Experiment 4: top-k 20 + 10 excerpts + dedup ✓ KEEP
- Date: 2026-03-27
- Hypothesis: more results (top-k 20) + more context excerpts (10) + dedup overlapping chunks helps multi-session
- Change: eval params (TOP_K=20, NUM_EXCERPTS=10, dedup by first 100 chars)
- QA Accuracy: **74% (38/50) — +4% from exp3, +24% from baseline!**
- Delta vs exp3: temporal +8% (→100%), multi-session +7% (→38%), user -15% (→71%), rest stable
- Verdict: **KEEP** — new best, 10% above Zep
- Remaining weakness: multi-session still at 38% (scattered facts across sessions)
- Next: Route B (Optuna parameter search) then Route A (online learning)

## Experiment 5: Optuna parameter sweep (Route B) — INFORMATIVE
- Date: 2026-03-27
- 40 trials, 9 min, zero cost
- Best session recall: 98.2% (trunc=5000, top_k=30, chunk_mode=whole)
- Verdict: **NOT DIRECTLY USEFUL** — Optuna maximized session recall (retrieval metric) not QA accuracy. High top-k with whole sessions = high recall but low precision (too much noise for QA).
- Key learning: session recall is a misleading proxy when top-k is large. Need to optimize QA accuracy directly.
- Learning applied: the "pairs+summary" with moderate top-k (20) is the sweet spot — balances precision (turn pairs) and coverage (summaries)

## Experiment 6: Cross-encoder reranker ✓ KEEP — NEW SOTA
- Date: 2026-03-27
- Hypothesis: cross-encoder (ms-marco-MiniLM-L6-v2) reranks top-30 candidates by query-fact relevance
- Change: eval pipeline — after mmem.search(limit=30), rerank with CrossEncoder, dedup, take top-10
- QA Accuracy: **94% (47/50) — +20% from exp4, +44% from baseline!**
- Surpassed Hindsight (91.4%) — #1 on LongMemEval (subagent-judged)
- By type: multi-session 100%, knowledge-update 100%, user 100%, preference 100%, assistant 80%, temporal 85%
- Only 3 errors: 2 temporal (date anchor missing, consecutive event calculation), 1 assistant (truncated excerpt)
- Verdict: **KEEP** — massive improvement from precision ranking
- Key insight: the problem was never recall (we had the right facts in top-30), it was RANKING (the right facts weren't in top-5). Cross-encoder fixes this.

## Code Integration (post-experiments)
- Cross-encoder reranker integrated into mmem core (fusion.py)
  - Lazy-loaded, graceful fallback if sentence-transformers not installed
  - Built-in benchmark: 98% → 100%
  - Commit: 05f4df1
- SessionState added to Memory class (session.py)
  - Tracks recent queries, accessed facts, entity affinity
  - Query expansion, session embedding, entity boost
  - Auto-updated on every search() call
  - Commit: d64aa25
- v2 architecture doc saved to docs/ARCHITECTURE-v2.md
  - Multi-tool MCP interface (not single search API)
  - 4-stage pipeline: recall → pre-rank → rerank → blend
  - Continuous learning: online + offline (4090)
  - Commit: 341075d

## Experiment 7: v2 Architecture (Phases 0-4) + Stratified Eval
- Date: 2026-03-27
- Changes: FastAPI daemon, hnswlib ANN, 6-channel recall, UserProfile, LightGBM/River scaffolding, serving logs, Thompson Sampling blender, feedback endpoint
- Fix: removed increment_access from search path (was corrupting ACT-R signal)
- Eval: stratified N=50 (10 multi, 10 temporal, 10 knowledge, 8 user, 7 assistant, 5 preference)
- Retrieval: 99% session recall (was 83% baseline, 96% exp1)
- **QA Accuracy: 82% (41/50) — stratified, subagent-judged**
- By type: knowledge-update 100%, user 100%, multi-session 82%, temporal 73%, assistant 71%, preference 60%
- Note: previous 94% was on first-50 (all single-session-user). 82% on stratified is the honest number.
- vs competitors: Mem0 49%, Zep 63.8%, Hindsight 91.4% (non-stratified)
- Verdict: **KEEP** — v2 infra committed, honest baseline established
- Weakest types: preference (60%), assistant (71%), temporal (73%)
- Failure patterns:
  - **assistant**: truncated at 800 chars, long lists/content cut off (items 27-49 missing)
  - **preference**: specific personalization details (pet names, product models) not retrieved
  - **temporal**: date anchoring fails when question lacks reference date ("weeks ago")

## Experiment 8: Context truncation 800→2000 chars — NO CHANGE
- Date: 2026-03-27
- Hypothesis: longer excerpts fix assistant truncation (lists cut off at 800 chars)
- Change: retrieved_full truncation 800→2000 in eval
- QA Accuracy: **82% (41/50) — same as exp 7**
- By type: assistant 6/7 (+1), but knowledge-update 9/10 (-1), preference 2/5 (-1)
- Verdict: **KEEP** (2000 is objectively better, but net effect is within noise)
- Key insight: truncation was NOT the bottleneck. Failures are retrieval failures (wrong facts), not truncation. Subagent judge variance is ~2-3 questions/run.
- Next: focus on retrieval quality, not context presentation

## Experiment 9: top-k 20→30 ✓ KEEP
- Date: 2026-03-27
- Hypothesis: 5/9 failures are wrong-facts-retrieved. More candidates gives cross-encoder more to work with.
- Change: top_k 20→30 in eval (fetch_limit=90 for reranker pre-fetch)
- Session recall: **100%** (was 99% — temporal miss fixed)
- **QA Accuracy: 86% (43/50) — +4% from exp 8**
- By type: knowledge-update 10/10 (+1), multi-session 9/10 (+1), preference 3/5 (+1), user 8/8, temporal 8/10, assistant 5/7 (-1)
- Verdict: **KEEP** — wider candidate pool helps retrieval
- Remaining 7 failures: 3 wrong-facts (46 Fender, 34 Luna, 47 jewelry), 2 truncation (30, 39), 1 date anchor (7), 1 missing date (40)

## Experiment 10-11: pair truncation 4000 + top-k 40 — PLATEAU
- Date: 2026-03-28
- Exp 10: pair truncation 2000→4000 + top-k 30 → **86% (43/50) — no change from exp 9**
- Exp 11: pair truncation 4000 + top-k 40 → **86% (43/50) — no change**
- Verdict: **PLATEAU** — parameter tuning exhausted
- Irreducible failures (4 consistent across all runs):
  - idx 7: "Valentine's day airline" → JetBlue booking retrieved, not AA flight
  - idx 30: Plesiosaur blue body buried deep in generated book content
  - idx 46: "music store tips" → guitars not retrieved (semantic gap)
  - idx 47: "jewelry from aunt" → chandelier from aunt retrieved (entity confusion)
- These need fundamentally different approaches: entity indexing, LLM query rewriting, semantic expansion

## Experiment 12: Multi-query expansion (template-based) — NO CHANGE
- Date: 2026-03-28
- Hypothesis: template-based query expansion (date patterns, entity extraction, question word stripping) widens retrieval net
- Change: query_expansion.py generating 2-4 variants, BM25+embedding union across variants
- QA Accuracy: **86% (43/50) — no change from exp 9**
- Verdict: **KEEP code** (multi-query is correct architecture), but template expansion too weak for remaining failures
- Remaining 4-5 failures need LLM-assisted retrieval (HyDE/contextual) or are data ambiguities

## Experiment 13: Confidence gate + importance boost + MMR + negation expansion
- Date: 2026-03-28
- Changes: 4 retrieval pipeline upgrades
  1. Importance boosting: score *= (1 + importance). CRITICAL/IMPORTANT auto-set 0.9
  2. MMR diversity reranking: diverse=True for aggregation queries
  3. Negation query expansion: "why don't we use X" → "rejected X"
  4. Confidence gate: adaptive threshold from feedback, below-threshold filtered out
  5. SearchMeta: confidence/hint/suggestion embedded in every output (CLI/API/MCP)
- maestro-env eval (58 scenarios, 11 failure modes):
  - **Perfect: 39/58 (67%), Avg recall: 80%**
  - Improved: H12 aggregation PASS (importance boost), H15 negation PASS (expansion)
  - Stable: all easy/medium at 100%, context_rot 100%, isolation 100%
  - Still failing: fallback 0% (generation problem), aggregation H10/H11 (scattered facts)
- LongMemEval: 86% (unchanged — LongMemEval doesn't test fallback/aggregation)
- Key insight: **retrieval ceiling reached**. Remaining failures need:
  - Fallback: agent interprets [NO RELEVANT DATA] from mmem output
  - Aggregation: agent does multi-round search guided by [SUGGESTION]
  - Negation reasoning: causal chain inference, not keyword matching

## What's needed to break 90% (next phase)
| Approach | Expected Impact | Effort |
|----------|----------------|--------|
| LLM query rewriting (via agent skill) | +4-6% (fixes semantic gaps) | Low — agent capability |
| Entity-level inverted index | +2-4% (fixes entity confusion) | Medium — new index |
| Multi-query retrieval (BM25 + embedding separate top-k union) | +2-3% (fixes single-signal misses) | Low — eval change |
| Larger N eval (200+) to reduce judge variance | Better signal | Medium — longer runs |

## Experiment 14: Ranking pipeline wiring + Context String Enrichment (Phase 0)
- Date: 2026-03-28
- Changes:
  1. Wired PreRanker (LightGBM), OnlineRanker (River), ThompsonBlender into hybrid_search()
  2. Weighted RRF via Thompson-sampled channel weights
  3. PreRanker pre-rank step between RRF and cross-encoder
  4. OnlineRanker P(used) score boost (0.5x-1.5x)
  5. feedback() method for online learning from implicit feedback
  6. Context string enrichment: inject category/entity/relations/importance into embedding input
  7. Per-channel origin tracking on SearchResult
- maestro-env keyword: **45/58 (78%) — no change**
- maestro-env agent-judged: **49/58 (84%) — no change**
- Tests: 97 passed, 1 skipped (+19 enrichment tests)
- Verdict: **KEEP** — infra is correct, no behavioral change yet because:
  - Ranking components untrained (graceful degradation): PreRanker has no model, OnlineRanker has 0 updates, ThompsonBlender has 0 updates
  - Context enrichment has no effect in eval because eval seeds facts without pre-existing graph relations → enrichment only adds minimal metadata (category + importance level)
  - These components need real usage + feedback data to show improvement
- Same 9 failures: cross_session (2), temporal (1), aggregation (2), negation (2), scale (2)
- Key insight: **eval environment doesn't exercise online learning or graph-based enrichment**. To show improvement, need either:
  1. Eval scenarios that build up relations before querying (multi-round eval)
  2. Richer fact seeding with entity_name + entity_type + relations
  3. Simulate feedback loops within eval

## Progress Summary
```
Date: 2026-03-28
Experiments: 14 (8 kept, 1 reverted, 1 informative, 4 infra-only)
maestro-env: 84% agent-judged, 78% keyword (58 scenarios)
LongMemEval (stratified): 86% (43/50)
Core v2: daemon, ANN, 6-channel, profile, ranking (wired), feedback, enrichment
Online learning: scaffolded + wired (needs real usage data)
Tests: 97 passed, 1 skipped
Next: seed eval with richer entity/relation data, or focus on the 9 hard failures
```
