from __future__ import annotations

import asyncio
from typing import Optional

import typer

from maestro_memory.core.memory import Memory
from maestro_memory.ingestion.consolidate import ConsolidateResult, consolidate, expand_paths

# ── CLI 命令 ────────────────────────────────────────────────────


def consolidate_cmd(
    paths: list[str] = typer.Argument(..., help="File paths or glob patterns to consolidate"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project name for scoped memory"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be processed without writing"),
) -> None:
    """Batch-ingest files into memory with chunking and dedup."""
    files = expand_paths(paths)
    if not files:
        typer.echo("No files matched the given patterns.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(files)} file(s):")
    for f in files:
        typer.echo(f"  {f}")

    if dry_run:
        typer.echo("\n[dry-run mode]")

    result = asyncio.run(_run(files, project, dry_run))
    _print_summary(result)


# ── 异步入口 ────────────────────────────────────────────────────


async def _run(files: list, project: str | None, dry_run: bool) -> ConsolidateResult:
    mem = Memory(project=project)
    await mem.init()
    try:
        return await consolidate(mem, files, dry_run=dry_run)
    finally:
        await mem.close()


def _print_summary(r: ConsolidateResult) -> None:
    typer.echo(f"\nFiles processed:  {r.files_processed}")
    typer.echo(f"Chunks total:     {r.chunks_total}")
    typer.echo(f"Chunks skipped:   {r.chunks_skipped} (dedup)")
    typer.echo(f"Facts added:      {r.facts_added}")
    typer.echo(f"Entities created: {r.entities_created}")
    if r.errors:
        typer.echo(f"Errors: {len(r.errors)}")
        for e in r.errors:
            typer.echo(f"  {e}", err=True)
