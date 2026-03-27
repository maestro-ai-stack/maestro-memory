# mmem v2 Architecture — Continuous Learning Memory Engine

## Design Principles

1. **LLM is the brain, mmem is the memory** — LLM does query understanding, intent, reasoning. mmem does storage, retrieval, ranking, learning.
2. **Multi-interface, not single-call** — LLM freely combines mem_search/mem_profile/mem_graph/mem_timeline/mem_recent in multi-turn interactions.
3. **Implicit context, explicit optional** — mmem auto-tracks session state. LLM can optionally pass intent/entities for better results.
4. **Continuous learning** — every interaction generates training data. Periodic batch training on 4090. Online learning on CPU.
5. **Zero-infra base** — single SQLite file still works. Models are optional upgrades.

## MCP Tool Interface

LLM interacts with mmem via multiple MCP tools (not a single search API):

```
mem_search(query, intent?, entities?, limit?)
  → Search memory with optional hints from LLM

mem_profile()
  → User profile summary (entity affinities, recent topics, preferences)

mem_recent(n=10)
  → Last N accessed/added facts (session context)

mem_graph(entity, hops=2)
  → Entity relationship graph

mem_timeline(entity)
  → Chronological fact history for an entity (with validity windows)

mem_add(content, entity?, importance?)
  → Add new memory (RL model decides ADD/UPDATE/DELETE/NOOP)

mem_status()
  → Stats: fact count, entity count, last access, health
```

LLM decides which tools to use, in what order, how many times. mmem doesn't constrain.

## Internal Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    mmem Engine                           │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Session   │  │ User     │  │ Serving  │              │
│  │ State     │  │ Profile  │  │ Logger   │              │
│  │ (memory)  │  │ (sqlite) │  │ (sqlite) │              │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘              │
│        └──────┬──────┘──────┬──────┘                    │
│               ▼             ▼                            │
│  ┌──────────────────────────────────────────┐           │
│  │         Retrieval Pipeline                │           │
│  │                                           │           │
│  │  Multi-recall:                            │           │
│  │    BM25 + Embedding + Graph +             │           │
│  │    Entity-inverted + Time-window +        │           │
│  │    Personal (affinity-based)              │           │
│  │           ▼                               │           │
│  │  Pre-rank: LightGBM (15-dim features)    │           │
│  │           ▼                               │           │
│  │  Re-rank: Cross-encoder (BGE-reranker)   │           │
│  │           ▼                               │           │
│  │  Blend: LinUCB bandit (online learned)   │           │
│  └──────────────────────────────────────────┘           │
│               ▼                                          │
│  ┌──────────────────────────────────────────┐           │
│  │         Memory Management                 │           │
│  │                                           │           │
│  │  On add(): RL model (or BERT-tiny)       │           │
│  │    decides ADD / UPDATE / DELETE / NOOP   │           │
│  │                                           │           │
│  │  On search(): auto-update                 │           │
│  │    access_count, session_state,           │           │
│  │    entity_affinity, LinUCB weights        │           │
│  └──────────────────────────────────────────┘           │
│                                                          │
│  ┌──────────────────────────────────────────┐           │
│  │         Storage Layer                     │           │
│  │                                           │           │
│  │  SQLite: episodes, entities, relations,   │           │
│  │          facts, facts_fts, entities_fts,  │           │
│  │          serving_logs, user_profile       │           │
│  │                                           │           │
│  │  Model weights: ~/.maestro/memory/models/ │           │
│  │    intent.onnx, prerank.lgb,             │           │
│  │    rerank.onnx, memory_rl.onnx           │           │
│  └──────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

## Model Training Pipeline (4090)

```
Serving logs (auto-collected)
  ↓
Data pipeline (generate training data)
  ↓
Training jobs (SSH to 4090):
  mmem train intent      # DistilBERT, 5 min, daily
  mmem train prerank     # LightGBM, 1 min, daily
  mmem train rerank      # BGE-reranker finetune, 2h, weekly
  mmem train memory-rl   # Qwen2.5-3B GRPO, 2h, weekly
  ↓
Model registry (version + A/B test)
  ↓
Deploy to local inference (ONNX/torch)
```

### Training Data Sources

| Model | Positive signal | Negative signal | Cold start |
|-------|----------------|-----------------|------------|
| Intent | LLM auto-labels intent when calling mem_search | N/A | LongMemEval 50q |
| Pre-rank | facts used by LLM in response | returned but not used | LongMemEval features |
| Rerank | (query, fact) pairs where fact was used | random negatives | LongMemEval pairs |
| Memory-R1 | facts that were later UPDATE'd or kept | facts that were DELETE'd or never accessed | LongMemEval episodes |
| LinUCB | usage feedback (reward=1) | non-usage (reward=0) | No cold start needed |

## Online Learning (zero GPU, real-time)

| Component | Update trigger | Algorithm | Latency |
|-----------|---------------|-----------|---------|
| ACT-R activation | every search() | access_count++ | <0.1ms |
| Entity affinity | every search() | EMA update | <0.1ms |
| Session state | every search()/add() | FIFO + decay | <0.1ms |
| LinUCB blender | every search() | matrix update | <0.1ms |

## Deployment Modes

| Mode | Dependencies | Models | Expected QA |
|------|-------------|--------|-------------|
| **Lite** | sqlite + numpy | BM25 + MiniLM embedding + RRF | ~74% |
| **Standard** | + sentence-transformers + lightgbm | + BGE-M3 + LightGBM + cross-encoder + LinUCB | ~85% |
| **Full** | + ollama (3B model) | + Memory-R1 RL + query rewrite | ~90% |

## Metrics (corrected)

| Metric | What it measures | Role |
|--------|-----------------|------|
| Answer-Turn Recall | did we retrieve the turn with has_answer:true? | Retrieval quality |
| Precision@K | fraction of top-K that are relevant | Retrieval precision |
| nDCG@10 | position-weighted relevance | Ranking quality |
| QA Accuracy | LLM generates correct answer from context | End-to-end (primary) |
| Latency p50/p95 | search response time | Guard metric |

## Implementation Priority

1. SessionState + ServingLogger (core, local)
2. MCP Server (expose multi-tool interface)
3. BGE-M3 embedding upgrade (local)
4. LightGBM pre-rank (4090 train, local infer)
5. Cross-encoder rerank (4090 train, local infer)
6. LinUCB blender (local, online)
7. Memory-R1 RL (4090 train, local infer)
8. Continuous learning pipeline (train CLI)
