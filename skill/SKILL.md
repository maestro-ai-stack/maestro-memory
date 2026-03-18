---
name: maestro-memory
description: "Business knowledge graph for AI agents. Stores prospect intel, dataset experience, domain expertise, client interactions, and business decisions — NOT code architecture. Dual system with Claude auto-memory: auto-memory = collaboration HOW, mmem = business WHAT. Triggers: remember, recall, memory, search memory, prospect, dataset, client, domain knowledge, entity relationship, who works on, what did we learn, pricing, business decision."
---

# maestro-memory — Business Brain

mmem stores **business/domain knowledge** that dies with the session if not saved.
Claude auto-memory handles collaboration preferences — mmem handles everything else.

## The Split

| System | Stores | Example |
|--------|--------|---------|
| **Auto-memory** | HOW to collaborate | "don't summarize, I read diffs" |
| **mmem** | WHAT we know about the world | "Prof Srisuma needs auction data" |

## Mandatory Triggers

### WRITE triggers (`mmem add` before reply ends)

| Trigger | Entity type | Command |
|---------|-------------|---------|
| Learn prospect info | `prospect` | `mmem add --entity-name "Prof X" --entity-type prospect "research focus: auction theory, needs bidding data"` |
| Dataset lesson | `dataset` | `mmem add --entity-name "era5" --entity-type dataset "hourly data has gaps pre-1979"` |
| Business decision | `project` | `mmem add --entity-name "ra-data" --entity-type project --type decision "pricing: $800 → $1200 for L2 tier"` |
| Client interaction | `prospect` | `mmem add --entity-name "Prof X" --entity-type prospect "replied 2026-03-16, wants meeting next week"` |
| Domain insight | (general) | `mmem add --type observation "DiD with staggered treatment: use Sun & Abraham 2021"` |
| Entity relationship | — | `mmem relate "Prof X" "collaborates_with" "Prof Y" --subject-type prospect --object-type prospect` |

### READ triggers (`mmem search` before acting)

| Situation | Command |
|-----------|---------|
| About to draft cold email | `mmem search "prospect name"` |
| About to work on a dataset | `mmem search "dataset name"` |
| Discussing RA Data strategy | `mmem search "ra-data"` |
| User mentions prior decision | `mmem search "topic"` |
| Before research on a person | `mmem graph --entity "name"` |

## Entity Types

| Type | Use for | Example entities |
|------|---------|-----------------|
| `prospect` | Academic clients, potential clients | Prof Srisuma, Prof Huang |
| `dataset` | Data products, external datasets | era5, china-county-crosswalk |
| `project` | Business units, initiatives | ra-data, maestro-creative |
| `person` | Team members, collaborators | Li Ding, Ao Wang |
| `method` | Econometric/statistical methods | DiD, IV-2SLS, BLP |
| `tool` | Software, platforms | Supabase, GEE |

## CLI Reference

### Search
```bash
mmem search "query"                                        # hybrid (BM25+embedding+graph)
mmem search "query" --current                              # only valid facts
mmem search "query" --as-of 2026-03-01                     # point-in-time
mmem search "query" --limit 5                              # limit results
```

### Add
```bash
mmem add "fact"                                            # general observation
mmem add "fact" --entity-name "X" --entity-type prospect   # attach to entity
mmem add --type feedback "correction"                      # behavior correction
mmem add --type decision "choice" --importance 0.9         # high-importance decision
mmem add --file ./notes.md                                 # ingest file
```

### Relate
```bash
mmem relate "A" "predicate" "B" --subject-type prospect --object-type prospect
# predicates: collaborates_with, advises, needs_data, uses_method, works_at, depends_on
```

### Graph
```bash
mmem graph --entity "Prof X"      # entity + relations + neighbors
mmem graph --list-entities         # all entities
mmem graph --list-relations        # all relations
```

### Consolidate (batch ingest)
```bash
mmem consolidate <paths> --dry-run   # preview first
mmem consolidate <paths>             # execute
```

### Status
```bash
mmem status                          # DB stats
mmem config show                     # current config
```

## Do NOT Store

- Code architecture, API routes, schemas — **read the code**
- Git history, who-changed-what — **use git log**
- File paths, project structure — **use Glob/Grep**
- Rules already in CLAUDE.md — **no duplication**
- Ephemeral task state — **use Tasks/Plans**
- Tool descriptions already in CLAUDE.md §3 — **no duplication**

## Anti-patterns

```
BAD:  mmem add --entity-name "ra/suite" --entity-type project "API route: /api/ra/projects"
WHY:  Code-derivable. Read the file instead.

BAD:  mmem add "User prefers English by default"
WHY:  Already in CLAUDE.md PRE-FLIGHT. Use auto-memory for collaboration style.

GOOD: mmem add --entity-name "Prof Srisuma" --entity-type prospect "replied to cold email, asked about auction data, status: warm lead"
WHY:  Business intel that can't be read from any file.

GOOD: mmem add --entity-name "china-crop-phenology" --entity-type dataset "NBS yearbook pre-2000 has province-level only, county starts 2000"
WHY:  Dataset experience learned through painful debugging, not documented anywhere.
```
