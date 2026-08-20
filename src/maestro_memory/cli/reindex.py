from __future__ import annotations

import asyncio
from typing import Optional

import typer

from maestro_memory.core.memory import Memory
from maestro_memory.retrieval.embedding import NullEmbeddingProvider


def reindex_cmd(
    force: bool = typer.Option(False, "--force", help="Re-embed every fact, not just the unembedded ones"),
    batch: int = typer.Option(64, "--batch", help="Facts to commit per transaction"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project name"),
) -> None:
    """Backfill missing fact embeddings.

    A fact with no embedding is invisible to semantic recall while still
    appearing in keyword results, so the store looks healthy and quietly
    answers worse. Facts arrive unembedded whenever they were written while
    sentence-transformers was missing, or imported by a path that bypasses the
    ingestion pipeline. Use ``--force`` after changing the embedding model,
    since vectors from two different models are not comparable.
    """
    asyncio.run(_reindex(force, batch, project))


async def _reindex(force: bool, batch: int, project: str | None) -> None:
    mem = Memory(project=project)
    await mem.init()
    try:
        provider = mem._embedding_provider
        if provider is None or isinstance(provider, NullEmbeddingProvider):
            typer.echo(
                "No embedding provider available — nothing to write.\n"
                "Install it with: pip install 'maestro-memory[local]'"
            )
            raise typer.Exit(1)

        sql = "SELECT id, content FROM facts"
        if not force:
            sql += " WHERE embedding IS NULL"
        cur = await mem.store.db.execute(sql)
        rows = list(await cur.fetchall())
        if not rows:
            typer.echo("Every fact already carries an embedding.")
            return

        typer.echo(f"Embedding {len(rows)} facts with {type(provider).__name__}...")
        written = skipped = 0
        for start in range(0, len(rows), batch):
            for fact_id, content in rows[start:start + batch]:
                vector = await provider.embed(content or "")
                if vector is None:
                    skipped += 1
                    continue
                await mem.store.db.execute(
                    "UPDATE facts SET embedding = ? WHERE id = ?",
                    (vector.astype("float32").tobytes(), fact_id),
                )
                written += 1
            await mem.store.db.commit()
            typer.echo(f"  {min(start + batch, len(rows))}/{len(rows)}")

        typer.echo(f"Wrote {written} embeddings" + (f", skipped {skipped}" if skipped else ""))
        typer.echo("Restart any running server so the ANN index is rebuilt.")
    finally:
        await mem.close()
