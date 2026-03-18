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
    # Strip the home directory prefix that Claude Code uses for project hashing
    # e.g. -Users-alice-projects-myapp -> projects-myapp -> projects/myapp
    home_prefix = f"-{Path.home().as_posix().replace('/', '-')}-"
    name = project_dir.replace(home_prefix, "").replace("-", "/")
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
    """Ingest manually curated facts from a config file or inline list.

    To use: create a file at scripts/facts.py containing a list of tuples:
        FACTS = [
            # (entity_name, entity_type, fact_type, content, importance)
            ("my-project", "project", "decision", "Chose SQLite for storage", 0.8),
            ("some-tool", "tool", "observation", "Supports REST API", 0.5),
        ]

    Or define facts inline below for quick testing.
    """
    # Example facts for testing — replace with your own
    # facts = [
    #     ("my-project", "project", "decision", "Your project decision here", 0.8),
    #     ("some-tool", "tool", "observation", "Tool observation here", 0.5),
    # ]

    # Try loading from external config file first
    facts_file = Path(__file__).parent / "facts.py"
    if facts_file.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("facts", facts_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        facts = getattr(mod, "FACTS", [])
    else:
        print("[INFO] No scripts/facts.py found. Skipping session facts.")
        print("[INFO] Create scripts/facts.py with a FACTS list to ingest custom facts.")
        return 0

    count = 0
    for entity_name, entity_type, fact_type, content, importance in facts:
        if dry_run:
            print(f"[DRY] {fact_type:12s} | {entity_name:30s} | {content[:80]}")
        else:
            await mem.add(
                content,
                source_type="conversation",
                source_ref="bulk-ingest-session-facts",
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
