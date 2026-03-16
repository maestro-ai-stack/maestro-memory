# maestro-memory 2.0 — Cognitive-Inspired Agent Memory

## Design Philosophy

Human memory is not a database — it is a living system that encodes, consolidates,
retrieves, and forgets. maestro-memory models this process in three layers:

```
Working Memory    →  context window (LLM native, ~4 retrieved chunks)
Short-term Memory →  post-conversation auto-extract (synaptic consolidation)
Long-term Memory  →  periodic reorganization (systems consolidation)
```

Key principles:
1. **Zero infrastructure** — single SQLite file, no Neo4j, no API keys required
2. **Elaborative encoding** — structured extraction > raw storage
3. **Adaptive forgetting** — ACT-R activation decay, not infinite accumulation
4. **Bilingual** — Chinese + English first-class via BGE-M3 + jieba FTS5

## Architecture

```
                    ┌──────────────────────────────────┐
                    │        Working Memory             │
                    │   (context window, LLM native)    │
                    │                                   │
                    │  mmem search → inject ~4 chunks   │
                    └────────────┬─────────────────────┘
                                 │ retrieval boosts activation
                                 │
                    ┌────────────▼─────────────────────┐
                    │       Short-term Memory            │
                    │   (synaptic consolidation)         │
                    │                                   │
                    │  Stop hook → transcript extract    │
                    │  Pydantic structured output        │
                    │  3-tier dedup (hash/cosine/LLM)   │
                    │  Entity + relation extraction      │
                    └────────────┬─────────────────────┘
                                 │ periodic batch
                                 │
                    ┌────────────▼─────────────────────┐
                    │        Long-term Memory            │
                    │   (systems consolidation)          │
                    │                                   │
                    │  mmem consolidate (skill/cron)     │
                    │  Schema abstraction                │
                    │  Knowledge graph densification     │
                    │  Contradiction detection           │
                    │  ACT-R activation decay            │
                    └──────────────────────────────────┘
```

## Component Upgrades

| Component | v0.1 (current) | v2.0 (target) |
|-----------|---------------|---------------|
| Embedding | all-MiniLM-L6-v2 (22M, EN only) | BGE-M3 (568M, 100+ langs, triple retrieval) |
| FTS5 | unicode61 (no CJK) | jieba via wangfenjin/simple (CN word segmentation + pinyin) |
| Chunking | none (full text) | tiktoken 256-512 token fixed chunks |
| Extraction | free-form LLM prompt | Pydantic structured output (function calling) |
| Dedup | none | 3-tier: exact hash → cosine > 0.9 → LLM adjudication |
| Temporal | single (valid_from/valid_until) | bi-temporal (valid_at/invalid_at + created_at/expired_at) |
| Activation | Ebbinghaus simple decay | ACT-R: A = B(recency,freq) + w*S(cosine) + noise |
| OCR | none | GLM-OCR (0.9B, OmniDocBench #1) |
| Types | no py.typed | py.typed + pyright strict |

## Benchmark Strategy

### Framework
- supermemoryai/memorybench (pluggable, LoCoMo built-in)
- Baseline: filesystem + RAG = 74% (Letta benchmark)
- A/B eval via skill-creator (with_skill vs without_skill)

### Evaluation Scenarios

Domain-specific multi-turn extraction and recall:

| Scenario | Test Focus | Key Metrics |
|----------|-----------|-------------|
| **Multi-turn extraction** | Facts accumulate across turns, later queries recall earlier facts | precision@k, recall@k |
| **CRM extraction** | Customer name/company/deal/stage/next-action from sales calls | entity accuracy, relation completeness |
| **Calendar scheduling** | Dates, times, attendees, conflicts from conversational booking | temporal accuracy, conflict detection |
| **Travel planning** | Destinations, dates, preferences, budget constraints from chat | multi-hop recall, preference consistency |
| **Patient records** | Symptoms, diagnoses, medications, history from clinical notes | fact supersession, temporal queries |
| **HR candidate** | Skills, experience, interview feedback, offer status | entity resolution (same person different mentions) |
| **Paper management** | Title, authors, key findings, citations from research reading | semantic linking, graph traversal |
| **Research auto-search** | Track research questions, findings, dead-ends across sessions | episodic recall, knowledge graph growth |
| **Work projects (from chat)** | Extract project status, blockers, decisions from Slack/Lark messages | noisy input tolerance, multi-entity extraction |

### Metrics Per Scenario
- **Retrieval**: precision@5, recall@5, MRR, NDCG
- **Temporal**: point-in-time accuracy, fact supersession correctness
- **Entity**: resolution accuracy (merge/split), graph completeness
- **Efficiency**: latency (p50/p95), token cost, DB size

## Implementation Phases

### Phase 1 — consolidate command (make it work)
- [ ] `mmem consolidate` CLI command
- [ ] File scanner (memory/*.md + progress/*/notes.md + transcripts)
- [ ] tiktoken chunking (256 token, 10% overlap)
- [ ] Pydantic structured extraction (entities, facts, relations)
- [ ] 3-tier dedup
- [ ] Bi-temporal schema migration
- [ ] Unit tests

### Phase 2 — retrieval enhancement (make it good)
- [ ] BGE-M3 embedding provider
- [ ] jieba FTS5 (simple extension)
- [ ] ACT-R activation model
- [ ] py.typed + pyright strict
- [ ] Retrieval benchmarks (LoCoMo baseline)

### Phase 3 — automation + eval (make it real)
- [ ] Stop hook for short-term memory auto-extract
- [ ] GLM-OCR integration
- [ ] memorybench integration (all 9 scenarios)
- [ ] A/B eval via skill-creator
- [ ] consolidate as a skill (weekly cron)
- [ ] README update with benchmark results

## References

### Cognitive Science
- Atkinson & Shiffrin (1968) — Multi-store model
- Baddeley (2000) — Working memory with episodic buffer
- Tulving (1972) — Episodic vs semantic memory
- ACT-R activation: A = B(recency,freq) + w*S(cosine) + noise

### AI Memory Systems
- Graphiti (Zep) — bi-temporal knowledge graph, Pydantic extraction
- Mem0 — tool-calling extraction, cosine dedup
- A-MEM — Zettelkasten-inspired atomic notes with linking
- HippoRAG — neurobiologically inspired LTM for LLMs

### Tools
- BGE-M3: huggingface.co/BAAI/bge-m3
- wangfenjin/simple: github.com/wangfenjin/simple
- GLM-OCR: github.com/zai-org/GLM-OCR
- memorybench: github.com/supermemoryai/memorybench
