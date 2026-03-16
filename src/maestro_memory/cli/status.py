from __future__ import annotations

import asyncio
from typing import Optional

import typer

from maestro_memory.core.memory import Memory


def status_cmd(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project name"),
) -> None:
    """Show memory database statistics."""
    asyncio.run(_status(project))


async def _status(project: str | None) -> None:
    mem = Memory(project=project)
    await mem.init()
    try:
        stats = await mem.status()
        typer.echo(f"Database: {mem._db_path}")
        typer.echo(f"Entities:  {stats.get('entities', 0)}")
        typer.echo(f"Facts:     {stats.get('facts', 0)}")
        typer.echo(f"Relations: {stats.get('relations', 0)}")
        typer.echo(f"Episodes:  {stats.get('episodes', 0)}")
        size_kb = stats.get("db_size_bytes", 0) / 1024
        typer.echo(f"DB size:   {size_kb:.1f} KB")
    finally:
        await mem.close()
