# maestro-memory technical architecture

## Runtime boundary

`maestro-memory` is a local application. Its durable state is a SQLite file at
`~/.maestro/memory/<project>/mem.db`. The warm daemon binds to
`127.0.0.1:19830`; the CLI and Python client share that endpoint definition and
do not fall back to another service.

```text
agent / terminal
   ├─ mnerve compatibility CLI
   ├─ mmem CLI
   └─ local MCP process (optional)
             ↓
      loopback HTTP daemon
             ↓
        Memory facade
        ├─ ingestion
        ├─ retrieval
        ├─ ranking
        └─ SQLite store
```

The base package includes the local HTTP client and server. Embedding and
reranking models are optional; the SQLite/BM25 path remains usable without
them.

## Storage model

The core tables are:

- `episodes`: source observations and provenance;
- `facts`: retrievable semantic units with validity and importance;
- `entities`: named people, projects, datasets, methods, and concepts;
- `relations`: typed entity edges;
- `facts_fts` and `entities_fts`: SQLite FTS5 indexes;
- `serving_logs`: query results and usefulness feedback;
- `user_profile`: local affinity and recent interaction state.

Embeddings are stored as BLOBs in SQLite. No separate vector database is
required. Copying the database while the daemon is stopped is sufficient for a
backup or migration.

## Ingestion

`Memory.add()` records an episode, extracts or falls back to facts, resolves
entities, deduplicates facts, and updates indexes in one local store.

`mmem consolidate` adds file-oriented processing:

1. read supported files;
2. extract text, using optional local OCR for image-oriented inputs;
3. split long text into bounded chunks;
4. remove exact and semantic duplicates;
5. add facts with source provenance.

Inputs are stored as factual units. Callers can supply entity and fact metadata
explicitly, keeping extraction decisions inspectable and local.

## Retrieval

Search uses independent candidate channels followed by ranking:

1. SQLite FTS5/BM25 keyword retrieval;
2. local embedding similarity when embeddings exist;
3. entity graph retrieval;
4. time-window and entity-affinity retrieval;
5. feature pre-ranking and reciprocal-rank fusion;
6. temporal activation and diversity control;
7. optional local cross-encoder reranking.

Results carry the fact, linked entity, score, and search confidence metadata.
The engine can therefore degrade to BM25 without making recall unavailable.

## Temporal behavior

Facts have validity intervals and access metadata. Current searches exclude
invalidated facts; `as_of` searches evaluate the stored validity interval at a
past timestamp. Recency, access count, and explicit importance affect ranking
without overwriting lexical or semantic relevance.

## Feedback loop

`mnerve understand` emits a local query identifier and copyable `fact:<id>`
targets. `mnerve feedback` records which returned facts were used, or that none
were useful. Feedback updates serving logs, access counts, and entity affinity
inside the same database.

This makes observed agent use the training signal while preserving an
inspectable local audit trail.

## Local service

The daemon exists to keep the SQLite connection and optional models warm. Its
endpoint is defined once in `maestro_memory.server.config` and consumed by:

- daemon startup and health checks;
- the `mnerve` compatibility client;
- the async `MemoryClient`;
- the server command;
- the macOS launchd template.

HTTP clients disable environment proxy inheritance. The async client rejects
non-loopback endpoints, and the server entrypoint accepts only the configured
loopback host.

## MCP

The optional `mmem-mcp` command exposes local memory tools over stdio. It is an
alternate interface to the same engine and SQLite store, not a deployment
mode.

```bash
pip install "maestro-memory[mcp]"
mmem-mcp
```

## Verification

`./scripts/check.sh` is the single release gate. It runs:

1. the local-only boundary check;
2. Ruff over runtime and tests;
3. the complete pytest suite;
4. wheel and source-distribution builds;
5. package metadata validation.

The same command is used by the repository's pre-commit hook and GitHub CI.
