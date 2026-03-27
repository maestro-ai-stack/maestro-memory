# maestro-memory

**Remember everything. Retrieve what matters.**

[![PyPI version](https://img.shields.io/pypi/v/maestro-memory.svg)](https://pypi.org/project/maestro-memory/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Skills Ecosystem](https://img.shields.io/badge/skills-ecosystem-blueviolet)](https://github.com/anthropics/skills)

Cognitive-inspired memory for AI agents. Three-layer architecture modeled on human memory: working memory (context window), short-term memory (post-conversation extraction), and long-term memory (periodic consolidation with adaptive forgetting). Combines BM25 keyword search, embedding similarity, knowledge graph traversal, and ACT-R activation decay -- all in a single SQLite file. Zero infrastructure, zero config, zero API keys required.

---

## Quickstart

### For AI Agents

```bash
# Claude Code -- install as a skill (Vercel skills ecosystem)
npx skills add maestro-ai-stack/maestro-memory -y -g
```

Works with: **Claude Code** | **Cursor** | **Codex** | **Gemini CLI** | **OpenCode** | **Trae** and any agent that speaks CLI tools.

### For Developers

```bash
pip install maestro-memory
mmem add "User prefers snake_case for Python variables"
mmem search "coding style preferences"
```

Try it now:

```bash
$ mmem add "User prefers dark mode in all editors"
Episode #1 stored.
  facts added=1 updated=0 invalidated=0 entities=0

$ mmem add --type feedback "Do not use emoji in commit messages"
Episode #2 stored.
  facts added=1 updated=0 invalidated=0 entities=0

$ mmem search "editor preferences"
1. [0.0082] User prefers dark mode in all editors
   type=observation valid_from=2026-03-16 12:00:00

$ mmem status
Database: ~/.maestro/memory/default/mem.db
Entities:  0
Facts:     2
Relations: 0
Episodes:  2
DB size:   28.0 KB
```

---

## Why maestro-memory?

| | maestro-memory | mem0 | Letta | Graphiti |
|---|---|---|---|---|
| Retrieval | BM25 + embedding + graph + temporal | Embedding only | Embedding only | Graph + embedding |
| Temporal reasoning | Built-in validity windows + decay | No | No | Limited |
| Knowledge graph | SQLite relations, multi-hop traversal | No | No | Neo4j required |
| Infrastructure | Single SQLite file | Qdrant/PostgreSQL | PostgreSQL | Neo4j + PostgreSQL |
| API key required | No (local embeddings + BM25) | Yes | Yes | Yes |
| Zero-config | Yes | No | No | No |
| **LongMemEval QA** | **94%** | 49% | — | 63.8% (Zep) |

### Benchmark Results

Evaluated on [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025), the standard benchmark for long-term chat memory:

| System | QA Accuracy | Architecture |
|--------|-------------|-------------|
| **maestro-memory** | **94%** | BM25 + embedding + graph + cross-encoder rerank |
| Hindsight (Vectorize) | 91.4% | 4-network retain-recall-reflect |
| Zep / Graphiti | 63.8% | Temporal knowledge graph |
| Mem0 | 49% | Vector + graph + KV |

Key: cross-encoder reranking on top of multi-signal recall. See [TECHNICAL.md](docs/TECHNICAL.md) for the full 6-experiment autoresearch story.

---

## Architecture

```
Working Memory (context window, LLM native)
      |  mmem search --> inject ~4 chunks
      v
Short-term Memory (synaptic consolidation)
      |  Stop hook --> transcript extract --> mmem add
      v
Long-term Memory (systems consolidation)
      |  mmem consolidate --> chunk + dedup + extract + store
      v
  Memory Facade
      |
      +-- Ingestion Pipeline
      |     chunking (tiktoken 256t) --> dedup (hash+cosine)
      |     LLM extraction (optional) --> fallback (raw storage)
      |     OCR (GLM-OCR via ollama, optional)
      |
      +-- Query Engine
      |     1. BM25 (FTS5 + jieba)  --> sparse scores
      |     2. Embedding (BGE-M3)   --> dense scores
      |     3. Graph walk            --> relationship scores
      |     4. RRF fusion            + ACT-R activation decay
      |     5. Return top-k
      |
      +-- Storage Layer
            SQLite + FTS5 + numpy embeddings
            Tables: episodes | entities | relations | facts
            Single file: ~/.maestro/memory/{project}/mem.db
```

---

## CLI Reference

### Add memories

```bash
mmem add "User prefers snake_case"                       # store a fact
mmem add --type feedback "Do not mock databases in tests" # typed fact
mmem add --type decision "Deploy to Fly.io"              # decision
mmem add --file meeting-notes.md --source file           # ingest file
mmem add --importance 0.9 "Critical deadline: April 15"  # high importance
```

### Search memories

```bash
mmem search "coding preferences"                         # hybrid search
mmem search "tech stack" --current                       # only valid facts
mmem search "deadlines" --as-of 2026-03-01               # point-in-time
mmem search --entity "maestro-fetch" --hops 2            # graph neighborhood
mmem search "python" --limit 5                           # limit results
```

### Knowledge graph

```bash
mmem graph --entity "maestro-fetch"                      # entity details + relations
mmem graph --list-entities                               # list all entities
mmem graph --list-relations                              # list all relations
```

### Consolidate (batch ingest)

```bash
mmem consolidate ~/progress/*/notes.md                   # ingest markdown files
mmem consolidate ~/.claude/projects/*/memory/*.md         # ingest Claude memory files
mmem consolidate ~/docs/*.pdf ~/screenshots/*.png         # PDF + images (GLM-OCR)
mmem consolidate ~/data/ --dry-run                        # preview without writing
```

### Status and config

```bash
mmem status                                              # DB stats
mmem config init                                         # generate config.toml
mmem config show                                         # display config
```

---

## Python SDK

```python
from maestro_memory import Memory

mem = Memory()  # uses ~/.maestro/memory/default/mem.db
# or: mem = Memory(path="./project-memory.db")

await mem.init()

# Add
result = await mem.add("User prefers dark mode", source_type="conversation")
# result.facts_added, result.entities_created, result.episode_id

# Search
results = await mem.search("user preferences", limit=5)
for r in results:
    print(r.fact.content, r.score, r.fact.valid_from)

# Graph
data = await mem.graph("maestro-fetch", hops=2)
# data["entity"], data["relations"], data["neighbors"]

# Temporal
results = await mem.search("tech stack", current_only=True)
results = await mem.search("team members", as_of="2026-01-01")

# Status
stats = await mem.status()
# {"entities": 12, "facts": 47, "relations": 8, "episodes": 23, "db_size_bytes": 98304}

await mem.close()
```

---

## Installation

```bash
# Core -- BM25 + temporal + graph. No API key needed.
pip install maestro-memory

# Optional extras
pip install maestro-memory[local]     # Local embeddings (sentence-transformers)
pip install maestro-memory[openai]    # OpenAI embeddings + LLM extraction
pip install maestro-memory[anthropic] # Anthropic LLM extraction
pip install maestro-memory[all]       # Everything
```

### Development setup

```bash
git clone https://github.com/maestro-ai-stack/maestro-memory.git
cd maestro-memory
python3.11 -m venv .venv && source .venv/bin/activate
pip install hatchling && pip install -e ".[dev]"
pytest tests/ -v
```

---

## Configuration

Config lives at `~/.maestro/memory/config.toml`. Generate with `mmem config init`.

```toml
[embedding]
provider = "local"              # "local" | "openai" | "none"
model = "all-MiniLM-L6-v2"     # sentence-transformers model

[llm]
provider = "none"               # "openai" | "anthropic" | "none"

[storage]
base_dir = "~/.maestro/memory"
```

Storage layout:

```
~/.maestro/memory/
  default/           # global memory
    mem.db
  {project-hash}/    # per-project memory
    mem.db
  config.toml        # shared config
```

---

## Roadmap

### v0.2 — Retrieval Quality (current)

- [x] Cross-encoder reranking (ms-marco-MiniLM, +20% QA accuracy)
- [x] SessionState for implicit context tracking
- [x] LongMemEval benchmark infrastructure
- [ ] BGE-M3 embedding upgrade (384d → 1024d, multilingual)
- [ ] ANN index via sqlite-vss or hnswlib (scale to 100K+ facts)
- [ ] Contradiction detection (embedding similarity → LLM judge → invalidate)

### v0.3 — Service Architecture

- [ ] MCP server mode (persistent daemon, pre-loaded models, <10ms search)
- [ ] Multi-tool interface: `mem_search` / `mem_profile` / `mem_graph` / `mem_timeline` / `mem_recent`
- [ ] Serving logger (auto-collect training data from usage)
- [ ] Model hot-reload (swap models without restart)

### v0.4 — Learned Ranking

- [ ] LightGBM pre-ranker (15-dim features, trained from serving logs)
- [ ] LinUCB bandit blender (online-learned fusion weights, replaces static RRF)
- [ ] Query-dependent alpha (short query → BM25 weight up, long query → embedding up)
- [ ] Cross-encoder domain finetune on serving logs (4090 weekly)

### v0.5 — Continuous Learning

- [ ] `mmem train` CLI — trigger offline training on remote GPU
- [ ] Intent classifier (DistilBERT, daily retrain from usage data)
- [ ] Memory-R1 RL manager (Qwen2.5-3B GRPO for ADD/UPDATE/DELETE decisions)
- [ ] User profile embeddings (entity affinity + topic vectors)
- [ ] Periodic consolidation (merge low-activation facts, prune noise)

### v1.0 — Production

- [ ] Bi-temporal schema (assertion time vs event time)
- [ ] Multi-project memory sharing with ACL
- [ ] Streaming ingestion (watch files/conversations in real-time)
- [ ] OpenTelemetry metrics (search latency, recall, model performance)
- [ ] PyPI stable release

### Technical Evolution Map

```
                     Current              Next               Future
                     ───────              ────               ──────
Embedding       all-MiniLM-L6-v2    → BGE-M3 (1024d)    → domain-finetuned
                    (384d)              multilingual         on serving logs

Vector index    numpy linear scan   → sqlite-vss/hnswlib → FAISS with IVF
                    (<10K facts)        (<1M facts)          (unlimited)

Reranker        ms-marco-MiniLM     → BGE-reranker-v2-m3 → domain-finetuned
                    (English)           (multilingual)       cross-encoder

Fusion          RRF (static k=60)   → LinUCB bandit      → LambdaMART
                                        (online learned)     (listwise LTR)

Pre-rank        none                → LightGBM (15-dim)  → distilled from
                                                             cross-encoder

Memory mgmt     rule-based          → BERT-tiny           → Qwen2.5-3B
                    (fallback)          classifier (17MB)    GRPO (Memory-R1)

Context         SessionState        → MCP server          → agent feedback
                    (in-process)        (persistent)         loop (LinUCB)

Training        manual              → mmem train CLI      → continuous
                                        (SSH to 4090)        learning pipeline
```

Each upgrade is **backward-compatible** — every model is optional, and mmem always falls back to the previous tier. A single SQLite file remains the core.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, testing, and PR guidelines.

---

## License

MIT

---

Built by [Maestro](https://maestro.onl) — Singapore AI product studio.
