# maestro-memory: Temporal Hybrid Memory for Agents

**Date**: 2026-03-16
**Status**: Approved (autonomous design)
**Repo**: maestro-ai-stack/maestro-memory

## 1. Vision

maestro-memory is a lightweight, temporal-aware hybrid memory system for AI agents. It combines embedding search, BM25 keyword search, and knowledge graph traversal in a single SQLite file. Designed for developer tool agents (Claude Code, Cursor, Codex) that need to remember facts, track relationships, and reason about time — without requiring Neo4j, Qdrant, or any external infrastructure.

**Identity**: "Remember everything. Retrieve what matters. Forget what's stale."

**Key differentiators vs existing systems**:
- vs mem0: Built-in temporal reasoning + hybrid retrieval (not just vectors)
- vs Letta: Graph relationships + BM25 + no PostgreSQL needed
- vs Graphiti: Zero infrastructure (SQLite, not Neo4j)
- vs OpenViking: Graph relationships + temporal validity windows
- vs GraphRAG: Incremental (not batch), agent-native (not document-RAG)

## 2. Architecture

```
┌─────────────────────────────────────────────────┐
│                 maestro-memory                   │
├─────────────────────────────────────────────────┤
│  Interface Layer                                 │
│  ┌──────┐ ┌─────┐ ┌──────┐ ┌───────────────┐   │
│  │ CLI  │ │ SDK │ │Skill │ │ MCP (future)  │   │
│  └──┬───┘ └──┬──┘ └──┬───┘ └───────┬───────┘   │
│     └────────┼───────┘             │            │
│              ▼                     │            │
├─────────────────────────────────────────────────┤
│  Query Engine                                    │
│  ┌───────────────────────────────────────────┐   │
│  │ Input: natural language OR structured      │   │
│  │                                            │   │
│  │ 1. BM25 (FTS5) → sparse scores            │   │
│  │ 2. Embedding   → dense scores              │   │
│  │ 3. Graph walk  → relationship scores       │   │
│  │ 4. RRF fusion  + temporal decay            │   │
│  │ 5. Return top-k results                    │   │
│  └───────────────────────────────────────────┘   │
├─────────────────────────────────────────────────┤
│  Memory Store                                    │
│  ┌────────────────┐ ┌────────────────────────┐   │
│  │ Entities        │ │ Relations              │   │
│  │ - name, type    │ │ - subject → object     │   │
│  │ - summary       │ │ - predicate            │   │
│  │ - embedding     │ │ - valid_from/until     │   │
│  │ - updated_at    │ │ - confidence           │   │
│  └────────────────┘ └────────────────────────┘   │
│  ┌────────────────┐ ┌────────────────────────┐   │
│  │ Facts           │ │ Episodes               │   │
│  │ - content       │ │ - raw source text      │   │
│  │ - embedding     │ │ - timestamp            │   │
│  │ - valid_from    │ │ - source_type          │   │
│  │ - valid_until   │ │ (conversation, file,   │   │
│  │ - importance    │ │  git commit, etc.)     │   │
│  │ - entity_id     │ │                        │   │
│  └────────────────┘ └────────────────────────┘   │
├─────────────────────────────────────────────────┤
│  Storage: SQLite + FTS5 + numpy embeddings       │
│  Single file: ~/.maestro/memory/default/mem.db   │
└─────────────────────────────────────────────────┘
```

## 3. SQLite Schema

```sql
-- Episodes: raw source data (provenance)
CREATE TABLE episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT NOT NULL,
    source_type TEXT NOT NULL,  -- 'conversation' | 'file' | 'git' | 'manual'
    source_ref  TEXT,           -- file path, commit hash, session ID, etc.
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Entities: subjects in the knowledge graph
CREATE TABLE entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    entity_type TEXT NOT NULL DEFAULT 'concept',  -- 'person' | 'project' | 'tool' | 'concept'
    summary     TEXT NOT NULL DEFAULT '',
    embedding   BLOB,          -- numpy float32 array, serialized
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Relations: directed edges between entities with temporal validity
CREATE TABLE relations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id  INTEGER NOT NULL REFERENCES entities(id),
    predicate   TEXT NOT NULL,     -- 'uses' | 'prefers' | 'works_on' | 'depends_on' | ...
    object_id   INTEGER NOT NULL REFERENCES entities(id),
    confidence  REAL NOT NULL DEFAULT 1.0,
    valid_from  TEXT NOT NULL DEFAULT (datetime('now')),
    valid_until TEXT,              -- NULL = still current
    episode_id  INTEGER REFERENCES episodes(id),
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Facts: discrete statements with temporal validity and embeddings
CREATE TABLE facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT NOT NULL,
    fact_type   TEXT NOT NULL DEFAULT 'observation',  -- 'observation' | 'preference' | 'feedback' | 'decision'
    importance  REAL NOT NULL DEFAULT 0.5,            -- 0.0 to 1.0
    embedding   BLOB,
    entity_id   INTEGER REFERENCES entities(id),      -- primary entity this fact is about
    episode_id  INTEGER REFERENCES episodes(id),      -- provenance
    valid_from  TEXT NOT NULL DEFAULT (datetime('now')),
    valid_until TEXT,              -- NULL = still current
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Full-text search index for BM25
CREATE VIRTUAL TABLE facts_fts USING fts5(content, content=facts, content_rowid=id);
CREATE VIRTUAL TABLE entities_fts USING fts5(name, summary, content=entities, content_rowid=id);

-- Indexes
CREATE INDEX idx_relations_subject ON relations(subject_id);
CREATE INDEX idx_relations_object ON relations(object_id);
CREATE INDEX idx_facts_entity ON facts(entity_id);
CREATE INDEX idx_facts_valid ON facts(valid_from, valid_until);
CREATE INDEX idx_relations_valid ON relations(valid_from, valid_until);
```

## 4. Embedding Strategy

**Default: sentence-transformers (local, no API key)**
- Model: `all-MiniLM-L6-v2` (384 dims, 22MB, fast)
- Storage: numpy float32 arrays serialized as BLOB in SQLite
- Similarity: cosine similarity computed in Python (numpy dot product)
- Lazy loading: embeddings computed on first `add`, not on import

**Optional: OpenAI / Anthropic embeddings**
- Configure via `~/.maestro-memory/config.toml`
- `[embedding] provider = "openai"` or `"local"` (default)

**Why not sqlite-vss?** It adds a C extension dependency that breaks portability. numpy cosine on 10K facts takes <10ms — sufficient for agent memory sizes. If needed later, add Qdrant as optional backend.

## 5. Retrieval Pipeline

### 5.1 Hybrid Search

```python
async def search(query: str, *, limit: int = 10, current_only: bool = True,
                 as_of: str | None = None) -> list[SearchResult]:
    # 1. BM25 via FTS5
    bm25_results = fts5_search(query, limit=limit * 3)

    # 2. Embedding similarity
    query_embedding = embed(query)
    emb_results = cosine_top_k(query_embedding, all_fact_embeddings, k=limit * 3)

    # 3. Graph expansion (optional, if entities detected in query)
    entities_in_query = extract_entities(query)
    graph_results = graph_neighbors(entities_in_query, hops=2)

    # 4. RRF Fusion + Temporal Decay
    fused = reciprocal_rank_fusion(bm25_results, emb_results, graph_results)
    scored = apply_temporal_decay(fused, as_of=as_of, current_only=current_only)

    return scored[:limit]
```

### 5.2 Temporal Decay

```python
def temporal_score(fact, as_of=None):
    base = fact.importance
    now = as_of or datetime.utcnow()

    # Expired facts get zero score if current_only
    if fact.valid_until and fact.valid_until < now:
        return 0.0

    # Recency decay (Ebbinghaus-inspired)
    days_old = (now - fact.created_at).days
    recency = math.exp(-0.01 * days_old)

    # Access reinforcement
    access_boost = 1 + fact.access_count * 0.1

    return base * recency * access_boost
```

### 5.3 RRF Fusion

```python
def reciprocal_rank_fusion(*result_lists, k=60):
    scores = defaultdict(float)
    for results in result_lists:
        for rank, item in enumerate(results):
            scores[item.id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

## 6. Ingestion Pipeline

### 6.1 LLM Extraction (mem0-inspired)

```python
async def add(content: str, *, source_type: str = "manual",
              source_ref: str | None = None) -> AddResult:
    # 1. Store raw episode
    episode_id = store_episode(content, source_type, source_ref)

    # 2. LLM extracts structured memory operations
    operations = await llm_extract(content, existing_facts)
    # Returns: [
    #   {"op": "ADD", "fact": "User prefers snake_case", "entity": "user", "type": "preference"},
    #   {"op": "UPDATE", "fact_id": 42, "new_content": "..."},
    #   {"op": "INVALIDATE", "fact_id": 17, "reason": "contradicted by new info"},
    # ]

    # 3. Execute operations
    for op in operations:
        if op["op"] == "ADD":
            entity = get_or_create_entity(op["entity"])
            embedding = embed(op["fact"])
            store_fact(op["fact"], entity, episode_id, embedding, op["type"])
        elif op["op"] == "UPDATE":
            update_fact(op["fact_id"], op["new_content"])
        elif op["op"] == "INVALIDATE":
            invalidate_fact(op["fact_id"])  # sets valid_until = now

    return AddResult(episode_id=episode_id, operations=operations)
```

### 6.2 No-LLM Fallback

When no LLM API key is configured, skip extraction. Store the raw text as a single fact with embedding. BM25 + embedding search still works; just no entity/relation extraction.

## 7. CLI Interface

```bash
# Search
mmem search "user preferences for commit messages"
mmem search "tech stack" --current              # only non-invalidated facts
mmem search "project deadlines" --as-of 2026-03-01
mmem search --entity "maestro-fetch" --hops 2   # graph neighborhood

# Add
mmem add "User prefers snake_case for Python variables"
mmem add --type feedback "Do not use emoji in files"
mmem add --type project "Deadline moved to April 15" --source conversation
mmem add --file ./meeting-notes.md              # ingest file

# Graph
mmem graph --entity "maestro-fetch"             # show entity relationships
mmem graph --list-entities                       # list all entities
mmem graph --list-relations                      # list all relations

# Status
mmem status                                      # DB stats: entities, facts, episodes, size
mmem status --project                            # current project memory

# Config
mmem config init                                 # generate config.toml
mmem config show
```

## 8. Python SDK

```python
from maestro_memory import Memory

mem = Memory()  # uses ~/.maestro/memory/default/mem.db
# or: mem = Memory(path="./project-memory.db")

# Add
await mem.add("User prefers dark mode", source_type="conversation")

# Search
results = await mem.search("user preferences", limit=5)
for r in results:
    print(r.content, r.score, r.valid_from)

# Graph
neighbors = await mem.graph("maestro-fetch", hops=2)

# Temporal
results = await mem.search("tech stack", current_only=True)
results = await mem.search("team members", as_of="2026-01-01")

# Close
await mem.close()
```

## 9. Project Structure

```
maestro-memory/
├── pyproject.toml
├── LICENSE (MIT)
├── README.md
├── .claude/skills/memory/SKILL.md
├── .claude-plugin/plugin.json
├── .claude-plugin/marketplace.json
├── src/maestro_memory/
│   ├── __init__.py              # exports Memory class
│   ├── __main__.py              # python -m maestro_memory
│   ├── cli/
│   │   ├── __init__.py          # typer app
│   │   ├── search.py
│   │   ├── add.py
│   │   ├── graph.py
│   │   ├── status.py
│   │   └── config.py
│   ├── core/
│   │   ├── memory.py            # Memory class (main facade)
│   │   ├── store.py             # SQLite storage layer
│   │   ├── schema.py            # CREATE TABLE statements
│   │   ├── models.py            # dataclasses (Entity, Relation, Fact, Episode, SearchResult)
│   │   └── config.py            # TOML config loader
│   ├── retrieval/
│   │   ├── bm25.py              # FTS5 search wrapper
│   │   ├── embedding.py         # embedding provider (local/openai)
│   │   ├── graph.py             # graph traversal (recursive CTE)
│   │   ├── fusion.py            # RRF fusion + temporal decay
│   │   └── temporal.py          # temporal scoring, as-of filtering
│   └── ingestion/
│       ├── extractor.py         # LLM extraction pipeline
│       └── fallback.py          # no-LLM fallback (raw storage)
├── tests/
└── docs/
```

## 10. Dependencies

```toml
[project]
name = "maestro-memory"
version = "0.1.0"
dependencies = [
    "aiosqlite>=0.20",
    "numpy>=1.26",
    "typer>=0.12",
]

[project.optional-dependencies]
local = ["sentence-transformers>=3.0"]
openai = ["openai>=1.40"]
anthropic = ["anthropic>=0.34"]
all = ["maestro-memory[local]", "maestro-memory[openai]", "maestro-memory[anthropic]"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "ruff>=0.4"]

[project.scripts]
maestro-memory = "maestro_memory.cli:app"
mmem = "maestro_memory.cli:app"
```

## 11. Storage Location

```
~/.maestro/memory/
├── default/           # global memory
│   └── mem.db
├── {project-hash}/    # per-project memory
│   └── mem.db
└── config.toml        # shared config
```

Project hash = SHA-256 of absolute project path, truncated to 12 chars.

## 12. What We Ship in v0.1.0

**In scope**:
- SQLite storage with FTS5
- Embedding search (numpy cosine, local sentence-transformers)
- Graph traversal (recursive CTE)
- RRF fusion + temporal decay
- CLI (search, add, graph, status, config)
- Python SDK (Memory class)
- No-LLM fallback (store raw, search by BM25 + embedding)
- LLM extraction with OpenAI/Anthropic (optional)
- SKILL.md + plugin support
- Comprehensive tests

**Out of scope (future)**:
- MCP server
- Community memory sharing
- Qdrant/Neo4j backends
- Auto-compaction (summarizing old episodes)
- Multi-user / team memory
