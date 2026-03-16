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

## Contributing

Open issues and PRs on this repo. Run `ruff check` and `pytest tests/ -v` before submitting.

---

## License

MIT

---

Built by [Maestro](https://maestro.onl) -- Singapore AI product studio.
