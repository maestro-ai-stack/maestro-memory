---
name: maestro-memory
description: "Temporal hybrid memory for AI agents. Search, add, and traverse knowledge graphs with time-aware retrieval. Use when: agent needs to remember facts, track entity relationships, recall user preferences, query past decisions, or reason about what was true at a specific time. Triggers: remember, recall, memory, search memory, what did, user preference, project history, knowledge graph, entity relationship, temporal, as of, when did."
---

# maestro-memory Skill

Temporal hybrid memory system for AI agents. Combines BM25 keyword search, embedding similarity, and knowledge graph traversal with time-aware scoring.

## CLI Commands

### Search memory
```bash
mmem search "user preferences for commit messages"
mmem search "tech stack" --current          # only non-invalidated facts
mmem search "deadlines" --as-of 2026-03-01  # point-in-time query
mmem search "query" --entity "project" --hops 2  # graph neighborhood
```

### Add to memory
```bash
mmem add "User prefers snake_case for Python variables"
mmem add --type feedback "Do not use emoji in files"
mmem add --type project "Deadline moved to April 15" --source conversation
mmem add --file ./meeting-notes.md
```

### Explore knowledge graph
```bash
mmem graph --entity "maestro-fetch"    # show entity relationships
mmem graph --list-entities              # list all entities
mmem graph --list-relations             # list all relations
```

### Check status
```bash
mmem status            # DB stats: entities, facts, episodes, size
mmem status --project  # current project memory
```

### Configuration
```bash
mmem config init   # generate ~/.maestro/memory/config.toml
mmem config show   # display current config
```

## Python SDK

```python
from maestro_memory import Memory

mem = Memory()  # uses ~/.maestro/memory/default/mem.db
await mem.init()

# Add facts
await mem.add("User prefers dark mode", source_type="conversation")

# Search with hybrid retrieval
results = await mem.search("user preferences", limit=5)

# Graph traversal
neighbors = await mem.graph("maestro-fetch", hops=2)

# Temporal query
results = await mem.search("team members", as_of="2026-01-01")

await mem.close()
```

## When to Use

- **Remember**: Store user corrections, preferences, project decisions
- **Recall**: Search for past context, decisions, or preferences
- **Track**: Entity relationships and how they change over time
- **Temporal**: Query what was true at a specific point in time
- **Graph**: Explore connections between entities (projects, tools, people)

## Architecture

- Storage: single SQLite file with FTS5 for BM25
- Embeddings: local sentence-transformers (optional, graceful fallback)
- Graph: entity-relation triples with temporal validity windows
- Fusion: Reciprocal Rank Fusion + Ebbinghaus temporal decay
- LLM extraction: optional (OpenAI/Anthropic), falls back to raw storage
