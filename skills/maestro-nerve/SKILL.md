---
name: maestro-nerve
description: |
  MANDATORY local business knowledge layer. Stores and recalls facts about prospects, datasets, projects, decisions, methods, and client interactions.
  MUST search before drafting emails, working on datasets, discussing RA Data strategy, researching prospects, pricing, or client decisions.
  MUST store after learning durable prospect, dataset, business, domain, or client facts.
  Triggers: nerve, mnerve, remember, prospect, dataset, client, pricing, strategy, decision, RA Data, 知识图谱, 数据集, 客户, recall, understand.
allowed-tools: Bash(mnerve *)
---

# maestro-nerve

## First action

Run:

```bash
mnerve understand "$ARGUMENTS"
```

If no arguments were supplied, search for the current conversation topic.

## Runtime boundary

`mnerve` is the compatibility CLI for the consolidated local
`maestro-memory` service:

- launchd service: `com.maestro.memory`
- API: `http://127.0.0.1:19830`
- store: `~/.maestro/memory/default/mem.db`

It may start the native local Python daemon when unavailable. It never calls a
hosted memory service or starts a container runtime.

## Required loop

```text
1. SEARCH:   mnerve understand "topic"
2. FEEDBACK: mnerve feedback <query_id> fact:<id>
             use --none when no result was useful
3. DO:       complete the requested work
4. STORE:    mnerve remember "durable fact"
```

Search output is intentionally compact. Do not request or paste full JSON unless
debugging a specific field.

## Commands

```bash
mnerve understand "Ao Wang grant" --limit 3
mnerve feedback '<query_id>' fact:123
mnerve feedback '<query_id>' --none

mnerve remember "Ao asked about the NUS grant" --type fact --entity "Ao Wang" --entity-type person
mnerve remember "Proceed with proposal" --type decision --entity "Agentic AI grant" --entity-type project
mnerve status
```

Valid feedback targets are `fact:<id>`. `remember` accepts
`observation`, `fact`, `claim`, `conclusion`, `note`, `feedback`,
`preference`, and `decision`; compatibility kinds map to the local fact
ledger.

## Store

Store durable facts about clients, prospects, datasets, business decisions,
methods, and domain insights. Use a stable `--idempotency-key` for retried
events.

Do not store code structure, file paths, git history, task state, or
collaboration preferences.
