from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer

from maestro_memory.core.memory import Memory


def add_cmd(
    content: Optional[str] = typer.Argument(None, help="Content to remember"),
    fact_type: str = typer.Option("observation", "--type", "-t", help="Fact type: observation|preference|feedback|decision"),
    source: str = typer.Option("manual", "--source", "-s", help="Source type: manual|conversation|file|git"),
    importance: float = typer.Option(0.5, "--importance", "-i", help="Importance score 0.0-1.0"),
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

    asyncio.run(_add(text, fact_type, source, source_ref, importance, project))


async def _add(
    content: str, fact_type: str, source: str,
    source_ref: str | None, importance: float, project: str | None,
) -> None:
    mem = Memory(project=project)
    await mem.init()
    try:
        result = await mem.add(
            content,
            source_type=source,
            source_ref=source_ref,
            fact_type=fact_type,
            importance=importance,
        )
        typer.echo(f"Episode #{result.episode_id} stored.")
        typer.echo(f"  facts added={result.facts_added} updated={result.facts_updated} "
                    f"invalidated={result.facts_invalidated} entities={result.entities_created}")
    finally:
        await mem.close()
