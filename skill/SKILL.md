---
name: maestro-memory
description: "Cognitive-inspired memory for AI agents. Three-layer architecture: working memory (context), short-term (post-conversation), long-term (consolidation). Use when: agent needs to remember facts, build entity relationships, recall user preferences, query past decisions, batch-ingest files, or reason about what was true at a specific time. Triggers: remember, recall, memory, search memory, what did, user preference, project history, knowledge graph, entity relationship, temporal, as of, when did, consolidate, relate, who works on."
---

# maestro-memory

Cognitive-inspired memory with BM25 (CN+EN) + embedding + knowledge graph + ACT-R activation decay. Single SQLite file, zero API keys.

## Setup

### Install

```bash
pip install maestro-memory                    # from PyPI
# or from source:
pip install -e ~/maestro-memory               # editable local install
```

### Verify

```bash
mmem status                                   # should show DB path + stats
mmem add "test fact" && mmem search "test"    # round-trip check
```

### Optional: Stop hook for short-term memory

Add to `~/.claude/settings.json` under `hooks.Stop`:

```json
{
  "type": "command",
  "command": "bash ~/maestro-memory/scripts/memory-save.sh",
  "timeout": 10000
}
```

This auto-extracts conversation highlights after each session.

### Optional: BGE-M3 embeddings

```bash
pip install maestro-memory[local]             # installs sentence-transformers
# edit ~/.maestro/memory/config.toml:
# [embedding]
# model = "bge-m3"
```

### Optional: Chinese FTS5

```bash
pip install maestro-memory[chinese]           # installs jieba
# auto-detected, no config needed
```

---

## Two Usage Modes

### Mode 1: Short-term Memory (within conversation)

The agent uses `mmem` CLI via **Bash tool** during the conversation. This is the primary mode.

**When to search** (Bash tool → `mmem search`):
- User mentions prior conversations/decisions ("last time...", "didn't we...")
- User mentions unfamiliar project/person/abbreviation
- Before starting a new task — search for relevant background
- User explicitly asks to recall/remember something

**When to write** (Bash tool → `mmem add` / `mmem relate`):
- User gives behavior correction → `mmem add --type feedback "..."`
- Project milestone or decision → `mmem add --type project "..." --entity "name" --entity-type project`
- New person/tool/project discovered → `mmem add "..." --entity "X" --entity-type person`
- Entity relationship → `mmem relate "A" "predicate" "B"`
- Important fact with high priority → `mmem add "..." --importance 0.9`

**Automatic (Stop hook)**: If configured, `scripts/memory-save.sh` runs after each conversation, extracting the last ~2000 chars of assistant messages into memory as raw conversation type.

### Mode 2: Long-term Memory (batch consolidation)

Periodic deep scan of project files, notes, and session transcripts. Run manually or via cron.

**When to trigger** (user says "consolidate memory", "organize memory", "import notes"):

1. **Identify sources** — ask user or use defaults:
   ```bash
   # Claude Code file-based memories
   ~/.claude/projects/*/memory/*.md

   # Project progress notes
   ~/progress/*/notes.md

   # Session transcripts (if accessible)
   ~/.claude/sessions/*/transcript.jsonl
   ```

2. **Dry-run first** — always preview before writing:
   ```bash
   mmem consolidate --dry-run <paths...>
   ```

3. **Confirm** with user, then execute:
   ```bash
   mmem consolidate <paths...>
   ```

4. **Post-consolidation** — build entity graph from consolidated facts:
   ```bash
   # Agent reads mmem search results, identifies entities, builds graph:
   mmem relate "Project Alpha" "owned_by" "Sarah Chen" --subject-type project --object-type person
   mmem relate "Sarah Chen" "reports_to" "Li Ding" --subject-type person --object-type person
   ```

5. **Show stats**:
   ```bash
   mmem status
   mmem graph --list-entities
   mmem graph --list-relations
   ```

---

## CLI Reference

### Search
```bash
mmem search "coding preferences"                      # hybrid search (CN+EN)
mmem search "tech stack" --current                     # only non-invalidated facts
mmem search "deadlines" --as-of 2026-03-01             # point-in-time query
mmem search "query" --limit 5                          # limit results
```

### Add
```bash
mmem add "plain fact"                                                    # observation
mmem add "fact" --entity "Alice" --entity-type person                    # with entity
mmem add --type feedback "correction"                                    # behavior
mmem add --type project "decision" --importance 0.9                      # project
mmem add --type decision "tech choice"                                   # decision
mmem add --file ./notes.md                                               # file
```

### Relate
```bash
mmem relate "Alice" "works_at" "Acme" --subject-type person --object-type concept
mmem relate "Alpha" "depends_on" "API" --subject-type project --object-type tool
mmem relate "Alice" "leads" "Alpha" --subject-type person --object-type project
```

### Graph
```bash
mmem graph --entity "Alice"          # entity + relations + neighbors
mmem graph --list-entities           # all entities
mmem graph --list-relations          # all relations
```

### Consolidate
```bash
mmem consolidate ~/progress/*/notes.md                 # markdown files
mmem consolidate ~/.claude/projects/*/memory/*.md       # Claude memory files
mmem consolidate ~/docs/*.pdf ~/screenshots/*.png       # PDF + images (GLM-OCR)
mmem consolidate <paths> --dry-run                      # preview only
mmem consolidate <paths> --project myproject            # scoped DB
```

### Status
```bash
mmem status                          # DB stats
mmem config show                     # current config
```

---

## Do NOT Store

- Code patterns or file paths — read the code directly
- Current conversation temp state — use tasks
- Rules already in CLAUDE.md — no duplication
- Raw large files — use consolidate with chunking instead of add --file for big docs
