<p align="center"><img src=".github/maestro-logo.png" alt="Maestro" width="120" /></p>
<h1 align="center">maestro-memory</h1>
<p align="center"><strong>Local memory for AI agents.</strong></p>

`maestro-memory` stores agent knowledge in one SQLite file and retrieves it with
keyword, semantic, graph, and temporal signals. The `mnerve` command is the
compact compatibility interface used by Maestro agents; `mmem` exposes the
lower-level memory tools.

The runtime is local by construction:

- data stays in `~/.maestro/memory/`;
- the optional warm daemon binds only to `127.0.0.1:19830`;
- the CLI has no hosted-memory or container fallback;
- BM25 recall works without an account, API key, or model download.

## Install

Python 3.11 or newer is required.

```bash
pip install maestro-memory
```

For local embeddings and cross-encoder reranking:

```bash
pip install "maestro-memory[local]"
```

## Use `mnerve`

The daemon starts locally on first use.

```bash
mnerve remember "The client approved the revised scope" \
  --type decision \
  --entity "Project Atlas" \
  --entity-type project \
  --idempotency-key "email:message-id"

mnerve understand "What did Project Atlas approve?" --limit 3
mnerve feedback query:17 fact:42
mnerve status
```

`understand` returns a short `query:<id>` token and copyable `fact:<id>` targets.
`feedback` records which results were useful so ranking can improve from real
use. Give retried writes the same `--idempotency-key`; they resolve to the
original episode without duplicating facts.

## Use `mmem`

```bash
mmem add "User prefers snake_case" --type feedback
mmem search "coding preferences" --limit 5
mmem graph --entity "Project Atlas"
mmem status
```

Ingest files or directories:

```bash
mmem consolidate notes.md
mmem consolidate ./research-notes/
mmem consolidate ./documents/*.pdf
```

## Local service

Commands auto-start a background daemon when needed. On macOS it can also be
installed as a launch agent:

```bash
mmem server-install
mmem server-stop
mmem server-uninstall
```

The endpoint is fixed to `http://127.0.0.1:19830`. The client rejects non-local
endpoints.

## Storage and backup

```text
~/.maestro/memory/
  config.toml
  default/
    mem.db
  <project-hash>/
    mem.db
```

Back up or move a memory store by copying its `mem.db` file while the daemon is
stopped.

## Python API

```python
from maestro_memory import Memory

memory = Memory()
await memory.init()

await memory.add(
    "The pilot uses monthly observations",
    source_type="conversation",
    entity_name="Project Atlas",
    entity_type="project",
)

results = await memory.search("pilot observation grain", limit=5)
for result in results:
    print(result.fact.content, result.score)

await memory.close()
```

## How retrieval works

```text
query
  ├─ SQLite FTS5 / BM25
  ├─ local embeddings (optional)
  ├─ entity graph
  └─ temporal activation
        ↓
   rank fusion
        ↓
   local cross-encoder rerank (optional)
```

Facts retain provenance, validity windows, importance, access counts, and
entity links. Missing optional models reduce retrieval quality without making
the store unavailable.

See [Technical documentation](docs/TECHNICAL.md) for the schema and retrieval
pipeline.

## Agent skills

Reusable skill entrypoints are under `skills/`:

- `skills/maestro-memory/` for the `mmem` interface;
- `skills/maestro-nerve/` for the compact `mnerve` workflow.

## Development

```bash
git clone https://github.com/maestro-ai-stack/maestro-memory.git
cd maestro-memory
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
./scripts/check.sh
```

The same check runs in pre-commit and CI. See [CONTRIBUTING.md](CONTRIBUTING.md)
for contribution guidelines and [SECURITY.md](SECURITY.md) for private security
reports.

## License

MIT

<p align="center">Built by <a href="https://maestro.onl">Maestro</a>.</p>
