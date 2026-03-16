#!/usr/bin/env python3
"""Bulk ingest existing Claude auto-memory files into maestro-memory.

Reads all ~/.claude/projects/*/memory/*.md files (except MEMORY.md),
parses frontmatter (name, description, type), and stores each as a fact
with the appropriate entity and type.

Also ingests key facts from the current session's conversation.

Usage:
    python scripts/bulk_ingest.py
    python scripts/bulk_ingest.py --dry-run
"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from maestro_memory.core.memory import Memory


def parse_frontmatter(text: str) -> dict:
    """Parse YAML frontmatter from a memory markdown file."""
    match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
    if not match:
        return {"body": text.strip()}
    fm_text, body = match.group(1), match.group(2)
    fm = {}
    for line in fm_text.strip().splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            fm[key.strip()] = val.strip().strip("'\"")
    fm["body"] = body.strip()
    return fm


def extract_project_name(project_dir: str) -> str:
    """Extract a readable project name from the Claude project hash dir."""
    # -Users-ding-maestro-projects-maestro -> maestro-projects-maestro
    name = project_dir.replace("-Users-ding-", "").replace("-", "/")
    # Take last 2 segments
    parts = name.split("/")
    return "/".join(parts[-2:]) if len(parts) > 1 else parts[-1]


async def ingest_memory_files(mem: Memory, dry_run: bool = False) -> int:
    """Ingest all existing Claude auto-memory files."""
    memory_dir = Path.home() / ".claude" / "projects"
    count = 0

    for md_file in sorted(memory_dir.rglob("memory/*.md")):
        if md_file.name == "MEMORY.md":
            continue

        text = md_file.read_text(encoding="utf-8", errors="ignore")
        if len(text.strip()) < 10:
            continue

        fm = parse_frontmatter(text)
        name = fm.get("name", md_file.stem)
        description = fm.get("description", "")
        mem_type = fm.get("type", "observation")
        body = fm.get("body", text)

        # Map memory type to fact_type
        type_map = {"feedback": "feedback", "user": "preference", "project": "decision", "reference": "observation"}
        fact_type = type_map.get(mem_type, "observation")

        # Entity from project dir
        project_dir = md_file.parent.parent.name
        project_name = extract_project_name(project_dir)

        # Content: combine description and body (truncated)
        content = f"{name}: {description}" if description else name
        if body and body != content:
            content += f"\n{body[:500]}"

        if dry_run:
            print(f"[DRY] {fact_type:12s} | {project_name:30s} | {content[:80]}")
        else:
            await mem.add(
                content,
                source_type="file",
                source_ref=str(md_file),
                fact_type=fact_type,
                entity_name=project_name,
                entity_type="project",
                importance=0.7 if fact_type == "feedback" else 0.5,
            )
            print(f"[ADD] {fact_type:12s} | {project_name:30s} | {content[:80]}")
        count += 1

    return count


async def ingest_current_session_facts(mem: Memory, dry_run: bool = False) -> int:
    """Manually curated facts from the current session (maestro-fetch + maestro-memory design)."""
    facts = [
        # User preferences
        ("user", "person", "preference", "Prefers CLI + Skill over MCP for tool distribution", 0.8),
        ("user", "person", "preference", "Wants autonomous execution: 'do it yourself, I'll check results tonight'", 0.7),
        ("user", "person", "preference", "Always prefix replies with 'Yes Minister'", 0.9),
        ("user", "person", "preference", "English by default, Chinese only when explicitly asked", 0.8),
        ("user", "person", "preference", "Annotate hard words (IELTS 6+) inline with short explanation", 0.7),
        ("user", "person", "feedback", "Do not use MCP servers — use CLI + Skill instead", 0.9),

        # maestro-fetch decisions
        ("maestro-fetch", "project", "decision", "v0.2.0 architecture: src/ layout, pluggable browser backends (CDP/bb-browser/Cloudflare/Playwright), community source adapters, SQLite cache with TTL", 0.8),
        ("maestro-fetch", "project", "decision", "Distribution: pip install + npx skills add + /plugin marketplace add", 0.7),
        ("maestro-fetch", "project", "decision", "CLI design: noun-verb subcommands (mfetch fetch/source/session/cache/config)", 0.6),
        ("maestro-fetch", "project", "decision", "bb-browser is complementary, not competing — pluggable browser backend", 0.7),
        ("maestro-fetch", "project", "decision", "Community source adapters in separate repo: maestro-ai-stack/maestro-fetch-sources", 0.6),
        ("maestro-fetch", "project", "decision", "Removed all MCP servers from settings.json — CLI + Skill only", 0.8),
        ("maestro-fetch", "project", "observation", "Added CDP backend: mfetch --cdp connects to running Chrome port 9222", 0.6),

        # maestro-memory decisions
        ("maestro-memory", "project", "decision", "v0.1.0 architecture: SQLite single-file, FTS5 BM25 + numpy embeddings + graph traversal, RRF fusion + ACT-R temporal activation", 0.8),
        ("maestro-memory", "project", "decision", "Borrows from: Graphiti (temporal), mem0 (extraction), Letta (tiers), OpenViking (hierarchy), nano-graphrag (simplicity)", 0.7),
        ("maestro-memory", "project", "decision", "Zero-config: works without LLM API key (BM25 only) and without sentence-transformers", 0.7),
        ("maestro-memory", "project", "decision", "Chinese support: jieba segmentation for BM25, BGE-M3 for multilingual embeddings", 0.6),

        # Organization
        ("maestro-ai-stack", "project", "observation", "GitHub organization for open-source maestro tools: maestro-fetch, maestro-fetch-sources, maestro-memory", 0.6),

        # Tool relationships
        ("maestro-fetch", "project", "observation", "maestro-fetch = agent perception layer (fetch external data)", 0.7),
        ("maestro-memory", "project", "observation", "maestro-memory = agent memory layer (store, retrieve, temporal reasoning)", 0.7),

        # Research findings
        ("bb-browser", "tool", "observation", "Chrome extension + CDP, real browser login state, 100+ JS site adapters, by Epiral org", 0.5),
        ("Cloudflare Browser Rendering", "tool", "observation", "REST API, free tier 10min/day, /markdown and /crawl endpoints, natural anti-bot advantage", 0.5),
        ("MiroFish", "tool", "observation", "Multi-agent prediction engine by 郭航江, 27k stars, 30M RMB investment from Chen Tianqiao, uses Zep Cloud for agent memory", 0.5),
        ("OpenViking", "tool", "observation", "ByteDance filesystem paradigm for agent context, viking:// protocol, L0/L1/L2 tiered loading", 0.5),
        ("Graphiti", "tool", "observation", "Zep's temporal knowledge graph: dual timestamps, validity windows, fact invalidation, hybrid BM25+vector+graph retrieval", 0.5),
    ]

    count = 0
    for entity_name, entity_type, fact_type, content, importance in facts:
        if dry_run:
            print(f"[DRY] {fact_type:12s} | {entity_name:30s} | {content[:80]}")
        else:
            await mem.add(
                content,
                source_type="conversation",
                source_ref="session-2026-03-16-maestro-fetch-memory-design",
                fact_type=fact_type,
                entity_name=entity_name,
                entity_type=entity_type,
                importance=importance,
            )
            print(f"[ADD] {fact_type:12s} | {entity_name:30s} | {content[:80]}")
        count += 1

    return count


async def main():
    dry_run = "--dry-run" in sys.argv
    mem = Memory()
    await mem.init()

    print("=== Phase 1: Ingest existing Claude auto-memory files ===\n")
    count1 = await ingest_memory_files(mem, dry_run=dry_run)
    print(f"\n{'Would ingest' if dry_run else 'Ingested'} {count1} memory files.\n")

    print("=== Phase 2: Ingest current session key facts ===\n")
    count2 = await ingest_current_session_facts(mem, dry_run=dry_run)
    print(f"\n{'Would ingest' if dry_run else 'Ingested'} {count2} session facts.\n")

    print(f"=== Total: {count1 + count2} facts ===")

    if not dry_run:
        status = await mem.status()
        print(f"\nDB Status:")
        for k, v in status.items():
            print(f"  {k}: {v}")

    await mem.close()


if __name__ == "__main__":
    asyncio.run(main())
