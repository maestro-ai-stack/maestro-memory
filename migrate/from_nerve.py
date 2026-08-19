#!/usr/bin/env python3
"""Migrate the exported mnerve knowledge graph into maestro-memory.

Reads the JSONL produced by ``migrate/export_from_nerve.sh`` and writes facts,
entities and relations into a maestro-memory SQLite store.

Three details make this non-trivial:

1. **Full text lives in ``properties.content``, not ``name``.** mnerve truncates
   ``kg_nodes.name`` to 100 characters; 410 of 778 nodes have longer content (up
   to 1,025 chars) and every one is an exact prefix of the truncation. Migrating
   ``name`` would discard most of the text of those facts.

2. **193 nodes came *from* maestro-memory already** (``properties.source =
   'mmem_migration'``), so a naive load would duplicate them. Dedup runs on
   ``ingestion.dedup.content_hash`` — the same canonical hash the ingestion path
   uses — with the hash set built once rather than per insert, since
   ``is_duplicate`` costs a DB round-trip per existing fact.

3. **Facts are written verbatim.** ``Memory.add()`` routes through LLM
   extraction with a rewriting fallback; these facts were already curated, so
   the migration writes through ``Store.add_fact`` instead.

Usage:
    python migrate/from_nerve.py --dry-run
    python migrate/from_nerve.py --db ~/.maestro/memory/default/mem.db
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from maestro_memory.core.config import load_config  # noqa: E402
from maestro_memory.core.store import Store  # noqa: E402
from maestro_memory.ingestion.dedup import content_hash  # noqa: E402
from maestro_memory.retrieval.embedding import get_embedding_provider  # noqa: E402

DEFAULT_EXPORT = Path.home() / ".maestro/backups/nerve-export-20260818"

# mnerve node types that name a thing rather than state something. These become
# entities; every node also yields a fact carrying its text.
ENTITY_TYPES = {"person", "project", "concept", "entity", "dataset", "community"}


def fact_text(node: dict) -> str:
    """Full text of a node: properties.content when present, else the name."""
    props = node.get("properties") or {}
    content = props.get("content")
    name = node.get("name") or ""
    if isinstance(content, str) and len(content) >= len(name):
        return content
    return name


def to_store_ts(value: str | None) -> str | None:
    """Render a Postgres TIMESTAMPTZ in the format the store already uses.

    mmem writes timestamps with SQLite ``datetime('now')`` — naive UTC,
    space-separated — and ``filter_temporal`` compares them as *strings*. Mixing
    in offset-aware ISO values breaks that comparison (' ' sorts before 'T') and
    makes ``temporal_score`` subtract across naive and aware datetimes. Migrated
    rows therefore adopt the store's own format rather than their source format.
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def is_superseded(node: dict) -> bool:
    props = node.get("properties") or {}
    return bool(props.get("superseded")) or node.get("superseded_by") is not None


async def migrate(export_dir: Path, db: Path | None, dry_run: bool, limit: int | None) -> int:
    memories = [json.loads(x) for x in (export_dir / "memories.jsonl").read_text().splitlines() if x.strip()]
    edges = [json.loads(x) for x in (export_dir / "edges.jsonl").read_text().splitlines() if x.strip()]
    if limit:
        memories = memories[:limit]

    emb_cfg = load_config().get("embedding", {})
    provider = get_embedding_provider(
        provider=emb_cfg.get("provider", "local"),
        model=emb_cfg.get("model", "all-MiniLM-L6-v2"),
    )
    print(f"embedding provider: {type(provider).__name__} (model={emb_cfg.get('model')})")

    store = Store(db) if db else Store(Path.home() / ".maestro/memory/default/mem.db")
    await store.init()

    existing = await store.list_facts(limit=100_000, current_only=False)
    seen: set[str] = {content_hash(f.content) for f in existing}
    print(f"store already holds {len(existing)} facts")

    stats: Counter[str] = Counter()
    name_to_entity: dict[str, int] = {}

    episode_id = None
    if not dry_run:
        episode_id = await store.add_episode(
            f"mnerve migration from {export_dir.name}", "nerve-migration", str(export_dir),
        )

    for node in memories:
        text = fact_text(node)
        if not text.strip():
            stats["skipped_empty"] += 1
            continue
        h = content_hash(text)
        if h in seen:
            stats["skipped_duplicate"] += 1
            continue
        seen.add(h)

        node_type = node.get("node_type") or "fact"
        stats[f"type:{node_type}"] += 1

        if dry_run:
            stats["would_insert"] += 1
            continue

        entity_id = None
        if node_type in ENTITY_TYPES:
            ent_name = (node.get("name") or "")[:200]
            if ent_name:
                if ent_name in name_to_entity:
                    entity_id = name_to_entity[ent_name]
                else:
                    entity, created = await store.get_or_create_entity(ent_name, node_type)
                    entity_id = entity.id
                    name_to_entity[ent_name] = entity.id
                    stats["entities_created"] += int(created)

        emb = await provider.embed(text)
        emb_bytes = emb.astype("float32").tobytes() if emb is not None else None

        fid = await store.add_fact(
            text,
            fact_type=node_type,
            importance=float(node.get("importance") or 0.5),
            embedding=emb_bytes,
            entity_id=entity_id,
            episode_id=episode_id,
        )
        stats["inserted"] += 1

        # add_fact stamps valid_from/created_at with now(); restore mnerve's own
        # timestamps so temporal ranking sees the real history, and close out
        # facts mnerve had already marked superseded.
        created_at = to_store_ts(node.get("created_at"))
        valid_until = (
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if is_superseded(node) else None
        )
        if created_at:
            await store.db.execute(
                "UPDATE facts SET created_at = ?, valid_from = ?, valid_until = ? WHERE id = ?",
                (created_at, created_at, valid_until, fid),
            )
        elif valid_until:
            await store.db.execute("UPDATE facts SET valid_until = ? WHERE id = ?", (valid_until, fid))
        if valid_until:
            stats["marked_superseded"] += 1
    if not dry_run:
        await store.db.commit()

    # ── Edges ─────────────────────────────────────────────────
    # mmem relations are entity-to-entity, but only 14 of 458 mnerve edges join
    # two entity-type nodes; the graph is overwhelmingly fact-to-fact. Three
    # destinations rather than one, so 279 of 458 survive instead of 14:
    #   entity <-> entity   -> relations
    #   'about' fact->entity -> facts.entity_id (what that edge actually means)
    #   'co_selected'        -> feedback signal, belongs to the ranking layer;
    #                           left in the export for Phase 2 rather than
    #                           misfiled as a semantic relation here.
    for edge in edges:
        a, b = edge.get("from_name"), edge.get("to_name")
        etype = edge.get("edge_type") or "related"
        if not a or not b:
            continue
        if etype == "co_selected":
            stats["edges_deferred_co_selected"] += 1
            continue

        sid, oid = name_to_entity.get(a[:200]), name_to_entity.get(b[:200])

        if sid is not None and oid is not None:
            if dry_run:
                stats["would_add_relation"] += 1
                continue
            await store.add_relation(sid, etype, oid, confidence=float(edge.get("confidence") or 0.5))
            stats["relations_added"] += 1
            continue

        # One end is an entity: attach the fact on the other end to it.
        ent_id, fact_name = (oid, a) if oid is not None else (sid, b)
        if ent_id is None:
            stats["edges_unmappable"] += 1
            continue
        if dry_run:
            stats["would_attach_fact_to_entity"] += 1
            continue
        cur = await store.db.execute(
            "SELECT id FROM facts WHERE entity_id IS NULL AND content LIKE ? LIMIT 1",
            (fact_name[:100] + "%",),
        )
        row = await cur.fetchone()
        if row:
            await store.db.execute("UPDATE facts SET entity_id = ? WHERE id = ?", (ent_id, row[0]))
            stats["facts_attached_to_entity"] += 1
        else:
            stats["edges_unmappable"] += 1

    if not dry_run:
        await store.db.commit()
    await store.close()

    print("\n" + ("DRY RUN — nothing written" if dry_run else "Migration complete"))
    for k in sorted(stats):
        print(f"  {k:32} {stats[k]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    ap.add_argument("--db", type=Path, help="target store (default: the live default store)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, help="only process the first N nodes")
    args = ap.parse_args()
    if not args.export.exists():
        sys.exit(f"export dir not found: {args.export}")
    return asyncio.run(migrate(args.export, args.db, args.dry_run, args.limit))


if __name__ == "__main__":
    sys.exit(main())
