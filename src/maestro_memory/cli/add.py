from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer

from maestro_memory.core.memory import Memory


async def _try_daemon_add(content: str, **kwargs) -> dict | None:
    """Try daemon first, return None if not available."""
    try:
        from maestro_memory.client import MemoryClient

        client = MemoryClient()
        await client.health()
        result = await client.add(content, **kwargs)
        await client.close()
        return result
    except Exception:
        return None


def add_cmd(
    content: Optional[str] = typer.Argument(None, help="Content to remember"),
    fact_type: str = typer.Option("observation", "--type", "-t", help="Fact type: observation|preference|feedback|decision"),
    source: str = typer.Option("manual", "--source", "-s", help="Source type: manual|conversation|file|git"),
    importance: float = typer.Option(0.5, "--importance", "-i", help="Importance score 0.0-1.0"),
    entity: Optional[str] = typer.Option(None, "--entity", "-e", help="Attach to entity (auto-created if new)"),
    entity_type: str = typer.Option("concept", "--entity-type", help="Entity type: person|project|tool|concept"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Ingest content from file"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project name"),
) -> None:
    """Add content to memory."""
    if file:
        if not file.exists():
            typer.echo(f"File not found: {file}", err=True)
            raise typer.Exit(1)
        text = file.read_text()
        source_ref = str(file)
    elif content:
        text = content
        source_ref = None
    else:
        typer.echo("Provide content or --file", err=True)
        raise typer.Exit(1)

    asyncio.run(_add(text, fact_type, source, source_ref, importance, entity, entity_type, project))


async def _add(
    content: str, fact_type: str, source: str,
    source_ref: str | None, importance: float,
    entity_name: str | None, entity_type: str, project: str | None,
) -> None:
    # Try daemon first for simple adds (no project override)
    if not project:
        kwargs = {
            "fact_type": fact_type,
            "source_type": source,
            "importance": importance,
        }
        if source_ref:
            kwargs["source_ref"] = source_ref
        if entity_name:
            kwargs["entity_name"] = entity_name
            kwargs["entity_type"] = entity_type
        daemon_result = await _try_daemon_add(content, **kwargs)
        if daemon_result is not None:
            typer.echo(f"Episode #{daemon_result['episode_id']} stored.")
            typer.echo(f"  facts added={daemon_result['facts_added']} updated={daemon_result.get('facts_updated', 0)} "
                        f"invalidated={daemon_result.get('facts_invalidated', 0)} entities={daemon_result['entities_created']}")
            return

    # Fallback to direct Memory access
    mem = Memory(project=project)
    await mem.init()
    try:
        result = await mem.add(
            content,
            source_type=source,
            source_ref=source_ref,
            fact_type=fact_type,
            importance=importance,
            entity_name=entity_name,
            entity_type=entity_type,
        )
        typer.echo(f"Episode #{result.episode_id} stored.")
        typer.echo(f"  facts added={result.facts_added} updated={result.facts_updated} "
                    f"invalidated={result.facts_invalidated} entities={result.entities_created}")
    finally:
        await mem.close()
