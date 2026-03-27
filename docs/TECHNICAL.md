# maestro-memory (mmem) Technical Documentation

## 1. Design Philosophy

### Zero-Infrastructure Memory

The entire memory system lives in a single SQLite file (`~/.maestro/memory/{project-hash}/mem.db`). No Redis, no Postgres, no Elasticsearch. This is a deliberate choice:

- **Portability**: copy one file to move your entire knowledge base.
- **Atomicity**: SQLite gives ACID transactions out of the box. No distributed coordination.
- **Latency**: in-process SQLite reads are sub-millisecond. The full search pipeline averages 6ms.

The database schema has five core tables (`episodes`, `entities`, `relations`, `facts`, `facts_fts`) plus two FTS5 virtual tables. Embeddings are stored as BLOBs directly in the `facts` and `entities` tables -- no separate vector store.

### Cognitive Science Grounding

The retrieval system draws from two bodies of cognitive science research:

**ACT-R (Adaptive Control of Thought -- Rational)**: John Anderson's theory of human memory activation. In ACT-R, the probability of recalling a memory chunk depends on how recently and frequently it was accessed, plus associative priming from the current context. mmem implements this directly in the temporal scoring function (see Section 2).

**Tulving's Memory Types**: The schema separates episodic memory (raw source episodes with provenance) from semantic memory (extracted facts and entity relationships). Episodes preserve the original context; facts are distilled, reusable knowledge units. This separation allows the same raw input to generate multiple facts linked to different entities, while maintaining provenance back to the source.

### Graceful Degradation

mmem works at three capability tiers, each adding models on top of the previous:

| Tier | Dependencies | What works |
|------|-------------|------------|
| **Bare** | sqlite3 + numpy | BM25 search, graph traversal, temporal scoring |
| **Embedded** | + sentence-transformers | + embedding search, cross-encoder reranking |
| **Full** | + LLM API key (OpenAI/Anthropic) | + entity extraction, fact decomposition, UPDATE/INVALIDATE |

Every model dependency is lazy-loaded and wrapped in a try/except. If `sentence-transformers` is not installed, `NullEmbeddingProvider` returns `None` and the pipeline skips embedding search. If no LLM API key exists, `fallback_extract()` stores the raw content as a single fact with default importance 0.5. The system never crashes due to a missing model -- it just retrieves less precisely.

## 2. Retrieval Architecture

The search pipeline has six stages, executed in `hybrid_search()` (`retrieval/fusion.py`):

```
Query
  |
  v
[1] BM25 (FTS5)  ──┐
[2] Embedding       ├──> [4] RRF Fusion ──> [5] Temporal Score ──> [6] Cross-Encoder Rerank
[3] Graph Traversal ┘
```

### Stage 1: BM25 via SQLite FTS5

FTS5 is SQLite's built-in full-text search extension. We use it instead of external search engines (Whoosh, Tantivy, Elasticsearch) because:

- It lives inside the same SQLite file -- zero extra infrastructure.
- It implements Okapi BM25 natively via the `bm25()` auxiliary function.
- It supports content tables (`content=facts, content_rowid=id`), so the FTS index stays synchronized with the source table.

**Query processing** (`retrieval/bm25.py`):

1. **Chinese segmentation**: if the query contains CJK characters and `jieba` is installed, it runs `jieba.cut_for_search()` for finer-grained tokenization. English text passes through as-is.
2. **FTS5 escaping**: strip special characters, split into tokens.
3. **Synonym expansion**: a lightweight synonym table maps abbreviations and related terms (e.g., `"bp"` -> `["blood", "pressure", "hypertension"]`). Expanded tokens are OR-joined.
4. **Scoring**: FTS5's `bm25()` returns negative scores (lower = better match). We negate them so higher = better.

Both facts and entities have separate FTS5 virtual tables (`facts_fts`, `entities_fts`), searched independently.

### Stage 2: Embedding Search

**Provider abstraction** (`retrieval/embedding.py`):

```python
class EmbeddingProvider(abc.ABC):
    async def embed(self, text: str) -> ndarray | None: ...
```

Three implementations:
- `NullEmbeddingProvider`: returns `None`. Used when no model is available.
- `LocalEmbeddingProvider`: wraps `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast, English-optimized).
- `BGEEmbeddingProvider`: wraps `BAAI/bge-m3` (1024-dim, multilingual, better for Chinese+English mixed content).

All providers are lazy-loaded -- the model is not instantiated until the first `embed()` call.

**Search**: the query embedding is compared against all stored fact embeddings via brute-force cosine similarity (`cosine_top_k()`). This is O(n) in the number of facts. For the current scale (hundreds to low thousands of facts), this completes in under 1ms. For larger scale, an approximate nearest neighbor index (FAISS, Annoy) would be needed.

### Stage 3: Graph Traversal

Entity-based recall exploits the knowledge graph structure.

1. The query is searched against `entities_fts` to find matching entities (top 5).
2. From those seed entities, a BFS traversal walks the `relations` table up to `hops=2` deep.
3. All facts linked to discovered entities are collected, with scores decaying by hop distance: `score = 1.0 / (hop + 1)`.

This enables associative recall -- asking about "Prof X" also surfaces facts about their collaborators, datasets they use, and projects they are part of, even if the query text does not mention those things.

### Stage 4: RRF Fusion

Reciprocal Rank Fusion merges the three ranked lists into a single ranking:

```
RRF_score(item) = sum over lists L:  1 / (k + rank_L(item) + 1)
```

where `k=60` is the smoothing constant.

**Why RRF and why k=60**: RRF was introduced by Cormack et al. (2009) as a rank-based fusion method that does not require score normalization across sources. BM25 scores, cosine similarities, and graph hop scores are on completely different scales -- RRF sidesteps this by using only rank positions. The constant `k=60` is the original paper's recommended value; it controls how much the fusion favors items ranked highly in multiple lists versus dominating a single list. Higher k means more weight to items that appear across multiple sources.

### Stage 5: ACT-R Temporal Scoring

After RRF fusion, each candidate fact is scored by the ACT-R activation model (`retrieval/temporal.py`):

```
A = B(recency, frequency) + w * S(similarity)

where:
  B = ln(access_count + 1) - 0.5 * ln(days_old + 1)    # base-level activation
  S = similarity                                          # spreading activation from context
  w = 0.3                                                 # spreading activation weight

final_score = importance * sigmoid(A)
```

**Base-level activation** combines two signals:
- **Frequency**: `ln(access_count + 1)` -- facts accessed more often have higher activation. The logarithm prevents runaway scores.
- **Recency**: `-0.5 * ln(days_old + 1)` -- activation decays with time, following a power law (linear in log-space). The 0.5 coefficient means activation halves roughly every e^2 days (~7.4 days).

**Spreading activation** (`w * S`): not yet wired in the current implementation (similarity defaults to 0.0), but the architecture supports boosting facts that are semantically close to the current query context.

**Importance weighting**: the sigmoid output (0 to 1) is multiplied by the fact's `importance` field (0.0 to 1.0), set at ingestion time. This allows high-importance facts to persist in retrieval even as they age.

The final search score is `rrf_score * temporal_score`, blending rank-based relevance with cognitive activation.

### Stage 6: Cross-Encoder Reranking

The single biggest accuracy improvement in the system. After stages 1-5 produce a candidate set (up to `limit * 5` items when reranking is enabled), a cross-encoder rescores every (query, fact) pair:

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(query, fact.content[:500]) for fact in candidates])
```

**Why cross-encoders matter**: BM25 and bi-encoder embeddings score query and document independently, then compare. A cross-encoder processes query and document jointly through all transformer layers, enabling fine-grained token-level attention. This is dramatically more accurate for relevance ranking but too expensive to run on the full corpus -- hence the two-phase approach (cheap recall, expensive rerank).

**Model choice**: `ms-marco-MiniLM-L-6-v2` is a 22M parameter model trained on MS MARCO passage ranking. It balances speed (~5ms per pair) and accuracy. The model is lazy-loaded and cached in a module-level global. If loading fails, the pipeline falls back to the RRF+temporal order.

**Implementation detail**: fact content is truncated to 500 characters before reranking to bound latency. The reranker sees more candidates than the final limit (5x) to maximize the chance that the truly relevant facts make it to the top.

## 3. Memory Lifecycle

### Write Path

```
User input (content string)
  |
  v
[1] Create episode (provenance record)
  |
  v
[2] Extract operations:
    - LLM available? -> llm_extract() -> structured ops (ADD/UPDATE/INVALIDATE)
    - No LLM?       -> fallback_extract() -> single ADD with raw content
  |
  v
[3] For each ADD operation:
    a. get_or_create_entity(name, type) -> entity linking
    b. embed(fact_content) -> float32 blob
    c. add_fact(content, embedding, entity_id, episode_id)
    d. Sync FTS5 index (jieba-segmented content)
  |
  v
[4] For UPDATE: store.update_fact(id, new_content) + FTS resync
    For INVALIDATE: store.invalidate_fact(id) -> sets valid_until = now
```

**LLM extraction** (`ingestion/extractor.py`): prompts gpt-4o-mini or claude-sonnet-4-20250514 with the content plus existing entity names, requesting a JSON array of operations. The prompt asks the LLM to decide: is this a new fact (ADD), an update to an existing fact (UPDATE), or does it contradict/supersede an existing fact (INVALIDATE)? This gives the system rudimentary belief revision without custom ML.

**Entity linking**: `get_or_create_entity()` does exact name matching. If the LLM extracts `"entity": "Prof X"`, the system finds or creates that entity. This is simple but effective -- the LLM normalizes names in its extraction step.

**Agent shortcut**: when the caller provides `entity_name` directly (e.g., `mmem add --entity "Prof X" "fact"`), LLM extraction is skipped entirely. The content is stored as-is with the specified entity. This is the common path for CLI usage.

### Read Path

```
Query string
  |
  v
[1] Multi-recall: BM25 + Embedding + Graph (parallel-ready, currently sequential)
  |
  v
[2] RRF fusion across recall sources
  |
  v
[3] Load facts, filter by temporal validity (current_only, as_of)
  |
  v
[4] ACT-R temporal scoring (recency + frequency + importance)
  |
  v
[5] Cross-encoder reranking (if available)
  |
  v
[6] Update access tracking (access_count++, last_accessed = now)
  |
  v
[7] Update session state (record_query, record_results)
```

### Manage Path

**Invalidation**: facts and relations have `valid_until` fields. Setting `valid_until = now` marks them as expired. They remain in the database (for historical queries with `as_of`) but are excluded from `current_only=True` searches (the default).

**Deduplication**: currently handled at the LLM extraction level -- the prompt includes existing entities so the LLM can reference them rather than creating duplicates. There is no automated deduplication of facts.

**Access tracking**: every fact returned by `hybrid_search()` gets `access_count += 1` and `last_accessed = now`. This feeds the ACT-R base-level activation, creating a natural reinforcement loop: frequently useful facts become easier to find.

## 4. Session Context

### SessionState

`SessionState` (`core/session.py`) is an ephemeral, in-memory object that tracks within-session context. It is not persisted to disk -- it resets when the `Memory` instance is closed.

```python
@dataclass
class SessionState:
    recent_queries: deque[str]           # last 20 queries (FIFO)
    recent_fact_ids: deque[int]          # last 50 fact IDs returned
    entity_activation: dict[int, float]  # entity_id -> activation score
    _session_embedding: ndarray | None   # EMA of query embeddings
```

### Query Expansion

`expand_query()` appends context words from the last 3 queries to the current query:

```
query = "hotel"
recent_queries = ["Tokyo trip planning", "budget for Japan"]
expanded = "hotel Tokyo trip planning budget Japan"
```

This helps follow-up queries that are terse or pronoun-heavy ("what about the hotel?" -> adds "Tokyo" and "Japan" from context). Only words longer than 2 characters are added, capped at 10 context words.

### Session Embedding

An exponential moving average (EMA) of query embeddings tracks the session's topic drift:

```
session_emb = decay * session_emb + (1 - decay) * query_emb
```

with `decay = 0.9`. This means the session vector is ~65% determined by the last 10 queries. It can be used to boost facts whose embeddings are close to the session topic (`similarity_to_session()`).

### Entity Activation

Entities gain activation when their facts appear in search results or when new facts are added:

```
On search result: activation = activation * 0.95 + 1.0
On add:           activation = activation * 0.95 + 2.0
```

The 0.95 multiplicative decay ensures old activations fade over the session. Entities repeatedly mentioned accumulate higher activation. `get_entity_boost()` returns the current activation level for use in retrieval scoring.

## 5. Benchmark Results

All experiments were run on LongMemEval_S (N=50 questions, subagent-judged) in a single day.

### Progression

| Experiment | Change | QA Accuracy | Delta |
|-----------|--------|-------------|-------|
| 0. Baseline | BM25 + embedding + RRF | 50% | -- |
| 1. More context | top-k 15, trunc 4000 | 50% | +0 (reverted) |
| 2. Session decomposition | turn pairs (~280 facts/q) | 66% | +16 |
| 3. + Session summaries | importance=0.7 summary chunks | 70% | +4 |
| 4. + top-k 20, dedup | more candidates, dedup by prefix | 74% | +4 |
| 5. Optuna sweep | 40 trials parameter search | -- | informative only |
| 6. Cross-encoder rerank | ms-marco-MiniLM-L-6-v2 | **94%** | +20 |

### Key Findings

**Experiment 2 (session decomposition)**: splitting raw sessions into individual turn pairs gave BM25 and embedding search much more precise targets. Single-session recall jumped +57-60%. The tradeoff: multi-session aggregation queries dropped -23% because facts became too granular.

**Experiment 3 (session summaries)**: adding full-session summary facts alongside turn pairs recovered multi-session performance (+8%) without hurting single-session gains.

**Experiment 5 (Optuna)**: maximizing session recall (retrieval metric) with 40 parameter trials achieved 98.2% recall -- but this was a misleading proxy. High top-k with whole sessions meant high recall but low precision, which did not translate to QA accuracy gains.

**Experiment 6 (cross-encoder)**: the largest single improvement (+20 points). The insight: the right facts were already in the top-30 candidates from RRF fusion. The problem was ranking -- the right facts were not in the top-5. Cross-encoder reranking fixed this precision bottleneck.

### Comparison with Other Systems

| System | QA Accuracy | Notes |
|--------|-------------|-------|
| Mem0 | 49% | LongMemEval paper reported |
| Zep | 63.8% | LongMemEval paper reported |
| Hindsight | 91.4% | Previous SOTA on LongMemEval |
| **mmem** | **94%** | Subagent-judged, same benchmark |

The 3 remaining errors (6%) were: 2 temporal reasoning failures (date anchor missing, consecutive event calculation) and 1 assistant-content truncation.

### Latency

Average search latency: **6ms** (without cross-encoder), measured on the built-in benchmark. Cross-encoder adds ~30-50ms depending on candidate count, still well under interactive thresholds.

## 6. Service Architecture (Planned)

### The Model Loading Problem

The current CLI workflow has a cold-start penalty: every `mmem search` invocation loads the embedding model (~3s) and the cross-encoder model (~2s) from disk. The actual search takes 6ms. This 500x overhead makes interactive use painful.

### MCP Server Mode

The planned solution is a persistent daemon exposed via the Model Context Protocol (MCP):

```
┌────────────┐     MCP (stdio/SSE)     ┌──────────────────┐
│ Claude Code ├────────────────────────>│  mmem MCP Server  │
│ (client)    │<────────────────────────│                    │
└────────────┘                          │  Pre-loaded:       │
                                        │  - embedding model │
┌────────────┐     MCP (stdio/SSE)     │  - cross-encoder   │
│ Another     ├────────────────────────>│  - SQLite conn     │
│ LLM client  │<────────────────────────│  - SessionState    │
└────────────┘                          └──────────────────┘
```

**Why MCP**: the protocol already has broad LLM client support (Claude Code, Cursor, Continue.dev). A multi-tool interface (`mem_search`, `mem_profile`, `mem_graph`, `mem_timeline`, `mem_recent`, `mem_add`, `mem_status`) gives the LLM freedom to compose retrieval strategies rather than being limited to a single search call.

### Multi-Tool Interface

The v2 architecture (see `docs/ARCHITECTURE-v2.md`) exposes seven MCP tools instead of a single search API:

- `mem_search(query, intent?, entities?, limit?)` -- hybrid search with optional LLM hints
- `mem_profile()` -- user/entity affinity summary
- `mem_recent(n)` -- last N accessed/added facts
- `mem_graph(entity, hops)` -- entity relationship traversal
- `mem_timeline(entity)` -- chronological fact history with validity windows
- `mem_add(content, entity?, importance?)` -- add new memory
- `mem_status()` -- database stats and health

The LLM decides which tools to call, in what order, how many times. mmem provides retrieval primitives; the LLM provides query understanding and reasoning.

### Online Learning Loop

Four components update in real-time on every search call:

| Component | Algorithm | Update trigger | Latency |
|-----------|-----------|---------------|---------|
| ACT-R activation | access_count++, recency | Every search result | <0.1ms |
| Entity affinity | EMA decay (0.95) + boost | Every search/add | <0.1ms |
| Session state | FIFO queues + EMA embedding | Every search/add | <0.1ms |
| LinUCB blender | Matrix update (planned) | Every search with feedback | <0.1ms |

**LinUCB**: an Upper Confidence Bound bandit algorithm for learning per-query blending weights across recall sources. Instead of fixed RRF fusion, LinUCB learns which sources are more reliable for different query types. Training signal: whether the LLM used a returned fact in its response (reward=1) or ignored it (reward=0).

### Offline Training Pipeline (Planned)

For heavier models, periodic batch training on a remote GPU (RTX 4090):

| Model | Task | Training time | Frequency |
|-------|------|--------------|-----------|
| LightGBM pre-ranker | 15-feature ranking before cross-encoder | 1 min | Daily |
| BGE-reranker finetune | Domain-adapted cross-encoder | 2h | Weekly |
| Memory-R1 (Qwen2.5-3B GRPO) | RL-based ADD/UPDATE/DELETE decisions | 2h | Weekly |
| DistilBERT intent classifier | Query intent routing | 5 min | Daily |

Training data comes from serving logs: which facts were returned, which the LLM actually used, which were added/updated/invalidated. This creates a flywheel where usage improves retrieval quality over time.
