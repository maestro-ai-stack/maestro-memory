---
name: maestro-memory
description: |
  Business knowledge brain — stores and retrieves facts about prospects, datasets, projects, decisions, domain expertise.
  AUTO-ACTIVATE on ANY non-trivial conversation: business strategy, prospect research, dataset work, domain discussion, cold email, pricing, product design, architecture decisions.
  Triggers: mmem, remember, recall, prospect, dataset, client, pricing, strategy, decision, research, 记忆, 知识图谱, 数据集, 客户.
  Do NOT use for: collaboration preferences (→ auto-memory), code patterns (→ read code), git history (→ git log).
allowed-tools: Bash(mmem *)
---

# maestro-memory — Business Brain

## First Action: Search Before Doing Anything

**When this skill loads, IMMEDIATELY run:**
```bash
mmem search "$ARGUMENTS"
```

If no arguments, search for the topic of the current conversation. Always search before acting — mmem may already know what you need.

## The Rule

**READ before act. WRITE before reply ends.**

```
Every conversation:
  1. SEARCH: mmem search "topic" — check what we already know
  2. DO: your actual work
  3. STORE: mmem add "new fact" — save anything learned
```

If you learn something about a person, dataset, business, or domain — and you don't `mmem add` it — that knowledge dies with the session.

## When to Search (mandatory)

| Situation | Command |
|-----------|---------|
| Start of any business conversation | `mmem search "topic"` |
| About to draft email to someone | `mmem search "person name"` |
| About to work on a dataset | `mmem search "dataset name"` |
| Discussing strategy/pricing/product | `mmem search "ra-data"` or `mmem search "maestro"` |
| User mentions a prior decision | `mmem search "topic"` |
| Before research on a person/org | `mmem graph --entity "name"` |

## When to Store (mandatory)

| Learned... | Command |
|------------|---------|
| Prospect info (name, field, needs, response) | `mmem add --entity "Prof X" --entity-type prospect "fact"` |
| Dataset lesson (quirk, gap, trick) | `mmem add --entity "era5" --entity-type dataset "fact"` |
| Business decision (pricing, strategy, kill) | `mmem add --entity "ra-data" --entity-type project --type decision "fact"` |
| Client interaction (replied, meeting, lost) | `mmem add --entity "Prof X" --entity-type prospect "fact"` |
| Domain insight (method, data source) | `mmem add "fact"` |
| Entity relationship | `mmem relate "A" "predicate" "B" --subject-type X --object-type Y` |

## The Split: mmem vs Auto-Memory

| System | Stores | Example |
|--------|--------|---------|
| **mmem** | WHAT we know about the world | "Prof Srisuma needs auction data" |
| **Auto-memory** | HOW to collaborate with user | "don't summarize, I read diffs" |

## Entity Types

| Type | Use for | Examples |
|------|---------|---------|
| `prospect` | Academic clients, potential clients | Prof Srisuma, Prof Huang |
| `dataset` | Data products, external datasets | era5, china-county-crosswalk |
| `project` | Business units, initiatives | ra-data, maestro-ai |
| `person` | Team members, collaborators | Ao Wang |
| `method` | Econometric/statistical methods | DiD, IV-2SLS, BLP |
| `tool` | Software, platforms | Supabase, GEE, SurrealDB |

## CLI Quick Reference

```bash
# Search (0.5s with daemon, auto-starts if needed)
mmem search "query"
mmem search "query" --limit 5
mmem graph --entity "name"

# Store
mmem add "fact"
mmem add "fact" --entity "X" --entity-type prospect
mmem add --type decision "choice" --importance 0.9
mmem relate "A" "predicate" "B" --subject-type prospect --object-type prospect

# Status
mmem status
```

## Do NOT Store

- Code architecture, API routes → **read the code**
- Git history → **git log**
- File paths → **Glob/Grep**
- Rules already in CLAUDE.md → **no duplication**
- Task state → **use Tasks/Plans**
