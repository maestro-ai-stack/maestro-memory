from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from maestro_memory.core.models import Entity, Episode, Fact, Relation
from maestro_memory.core.schema import get_schema
from maestro_memory.retrieval.tokenizer import segment


class Store:
    """SQLite storage layer for maestro-memory."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(get_schema())
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None, "Store not initialised — call init() first"
        return self._db

    # ── Episodes ──────────────────────────────────────────────────

    async def add_episode(self, content: str, source_type: str, source_ref: str | None = None) -> int:
        cur = await self.db.execute(
            "INSERT INTO episodes (content, source_type, source_ref) VALUES (?, ?, ?)",
            (content, source_type, source_ref),
        )
        await self.db.commit()
        return cur.lastrowid  # type: ignore[return-value]

    async def get_episode(self, episode_id: int) -> Episode | None:
        cur = await self.db.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,))
        row = await cur.fetchone()
        if not row:
            return None
        return Episode(id=row["id"], content=row["content"], source_type=row["source_type"],
                       source_ref=row["source_ref"], created_at=row["created_at"])

    # ── Entities ──────────────────────────────────────────────────

    async def add_entity(self, name: str, entity_type: str = "concept", summary: str = "",
                         embedding: bytes | None = None) -> int:
        cur = await self.db.execute(
            "INSERT INTO entities (name, entity_type, summary, embedding) VALUES (?, ?, ?, ?)",
            (name, entity_type, summary, embedding),
        )
        await self.db.commit()
        eid = cur.lastrowid
        # sync FTS (write after word segmentation)
        await self.db.execute(
            "INSERT INTO entities_fts (rowid, name, summary) VALUES (?, ?, ?)",
            (eid, segment(name), segment(summary)),
        )
        await self.db.commit()
        return eid  # type: ignore[return-value]

    async def get_entity(self, entity_id: int) -> Entity | None:
        cur = await self.db.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
        row = await cur.fetchone()
        return self._row_to_entity(row) if row else None

    async def get_entity_by_name(self, name: str) -> Entity | None:
        cur = await self.db.execute("SELECT * FROM entities WHERE name = ?", (name,))
        row = await cur.fetchone()
        return self._row_to_entity(row) if row else None

    async def get_or_create_entity(self, name: str, entity_type: str = "concept") -> tuple[Entity, bool]:
        """Return (entity, created). created=True if newly inserted."""
        existing = await self.get_entity_by_name(name)
        if existing:
            return existing, False
        eid = await self.add_entity(name, entity_type)
        entity = Entity(id=eid, name=name, entity_type=entity_type)
        return entity, True

    async def update_entity_summary(self, entity_id: int, summary: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "UPDATE entities SET summary = ?, updated_at = ? WHERE id = ?",
            (summary, now, entity_id),
        )
        # sync FTS
        await self.db.execute("DELETE FROM entities_fts WHERE rowid = ?", (entity_id,))
        cur = await self.db.execute("SELECT name FROM entities WHERE id = ?", (entity_id,))
        row = await cur.fetchone()
        if row:
            await self.db.execute(
                "INSERT INTO entities_fts (rowid, name, summary) VALUES (?, ?, ?)",
                (entity_id, segment(row["name"]), segment(summary)),
            )
        await self.db.commit()

    async def list_entities(self, limit: int = 100) -> list[Entity]:
        cur = await self.db.execute("SELECT * FROM entities ORDER BY updated_at DESC LIMIT ?", (limit,))
        rows = await cur.fetchall()
        return [self._row_to_entity(r) for r in rows]

    # ── Relations ─────────────────────────────────────────────────

    async def add_relation(self, subject_id: int, predicate: str, object_id: int,
                           confidence: float = 1.0, episode_id: int | None = None) -> int:
        cur = await self.db.execute(
            "INSERT INTO relations (subject_id, predicate, object_id, confidence, episode_id) VALUES (?, ?, ?, ?, ?)",
            (subject_id, predicate, object_id, confidence, episode_id),
        )
        await self.db.commit()
        return cur.lastrowid  # type: ignore[return-value]

    async def get_relations_for_entity(self, entity_id: int, current_only: bool = True) -> list[Relation]:
        sql = "SELECT * FROM relations WHERE (subject_id = ? OR object_id = ?)"
        params: list = [entity_id, entity_id]
        if current_only:
            sql += " AND valid_until IS NULL"
        cur = await self.db.execute(sql, params)
        rows = await cur.fetchall()
        return [self._row_to_relation(r) for r in rows]

    async def invalidate_relation(self, relation_id: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute("UPDATE relations SET valid_until = ? WHERE id = ?", (now, relation_id))
        await self.db.commit()

    # ── Facts ─────────────────────────────────────────────────────

    async def add_fact(self, content: str, fact_type: str = "observation", importance: float = 0.5,
                       embedding: bytes | None = None, entity_id: int | None = None,
                       episode_id: int | None = None) -> int:
        cur = await self.db.execute(
            "INSERT INTO facts (content, fact_type, importance, embedding, entity_id, episode_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (content, fact_type, importance, embedding, entity_id, episode_id),
        )
        await self.db.commit()
        fid = cur.lastrowid
        # sync FTS (write after word segmentation)
        await self.db.execute("INSERT INTO facts_fts (rowid, content) VALUES (?, ?)", (fid, segment(content)))
        await self.db.commit()
        return fid  # type: ignore[return-value]

    async def get_fact(self, fact_id: int) -> Fact | None:
        cur = await self.db.execute("SELECT * FROM facts WHERE id = ?", (fact_id,))
        row = await cur.fetchone()
        return self._row_to_fact(row) if row else None

    async def update_fact(self, fact_id: int, content: str) -> None:
        await self.db.execute("UPDATE facts SET content = ? WHERE id = ?", (content, fact_id))
        # sync FTS (write after word segmentation)
        await self.db.execute("DELETE FROM facts_fts WHERE rowid = ?", (fact_id,))
        await self.db.execute("INSERT INTO facts_fts (rowid, content) VALUES (?, ?)", (fact_id, segment(content)))
        await self.db.commit()

    async def invalidate_fact(self, fact_id: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute("UPDATE facts SET valid_until = ? WHERE id = ?", (now, fact_id))
        await self.db.commit()

    async def list_facts(self, limit: int = 100, current_only: bool = True) -> list[Fact]:
        sql = "SELECT * FROM facts"
        if current_only:
            sql += " WHERE valid_until IS NULL"
        sql += " ORDER BY created_at DESC LIMIT ?"
        cur = await self.db.execute(sql, (limit,))
        rows = await cur.fetchall()
        return [self._row_to_fact(r) for r in rows]

    async def increment_access(self, fact_id: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "UPDATE facts SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (now, fact_id),
        )
        await self.db.commit()

    # ── Stats ─────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        counts: dict[str, int] = {}
        for table in ("entities", "facts", "relations", "episodes"):
            cur = await self.db.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
            row = await cur.fetchone()
            counts[table] = row[0] if row else 0
        db_size = os.path.getsize(self.db_path) if self.db_path.exists() else 0
        return {**counts, "db_size_bytes": db_size}

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _row_to_entity(row: aiosqlite.Row) -> Entity:
        return Entity(id=row["id"], name=row["name"], entity_type=row["entity_type"],
                      summary=row["summary"], created_at=row["created_at"], updated_at=row["updated_at"])

    @staticmethod
    def _row_to_relation(row: aiosqlite.Row) -> Relation:
        return Relation(id=row["id"], subject_id=row["subject_id"], predicate=row["predicate"],
                        object_id=row["object_id"], confidence=row["confidence"],
                        valid_from=row["valid_from"], valid_until=row["valid_until"],
                        episode_id=row["episode_id"])

    @staticmethod
    def _row_to_fact(row: aiosqlite.Row) -> Fact:
        return Fact(id=row["id"], content=row["content"], fact_type=row["fact_type"],
                    importance=row["importance"], entity_id=row["entity_id"],
                    episode_id=row["episode_id"], valid_from=row["valid_from"],
                    valid_until=row["valid_until"], access_count=row["access_count"],
                    last_accessed=row["last_accessed"], created_at=row["created_at"])
