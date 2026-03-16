from __future__ import annotations

_SCHEMA = """
-- Episodes: raw source data (provenance)
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_ref  TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Entities: subjects in the knowledge graph
CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    entity_type TEXT NOT NULL DEFAULT 'concept',
    summary     TEXT NOT NULL DEFAULT '',
    embedding   BLOB,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Relations: directed edges between entities with temporal validity
CREATE TABLE IF NOT EXISTS relations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id  INTEGER NOT NULL REFERENCES entities(id),
    predicate   TEXT NOT NULL,
    object_id   INTEGER NOT NULL REFERENCES entities(id),
    confidence  REAL NOT NULL DEFAULT 1.0,
    valid_from  TEXT NOT NULL DEFAULT (datetime('now')),
    valid_until TEXT,
    episode_id  INTEGER REFERENCES episodes(id),
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Facts: discrete statements with temporal validity and embeddings
CREATE TABLE IF NOT EXISTS facts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    content       TEXT NOT NULL,
    fact_type     TEXT NOT NULL DEFAULT 'observation',
    importance    REAL NOT NULL DEFAULT 0.5,
    embedding     BLOB,
    entity_id     INTEGER REFERENCES entities(id),
    episode_id    INTEGER REFERENCES episodes(id),
    valid_from    TEXT NOT NULL DEFAULT (datetime('now')),
    valid_until   TEXT,
    access_count  INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Full-text search index for BM25
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(content, content=facts, content_rowid=id);
CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(name, summary, content=entities, content_rowid=id);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_id);
CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_id);
CREATE INDEX IF NOT EXISTS idx_facts_entity ON facts(entity_id);
CREATE INDEX IF NOT EXISTS idx_facts_valid ON facts(valid_from, valid_until);
CREATE INDEX IF NOT EXISTS idx_relations_valid ON relations(valid_from, valid_until);
"""


def get_schema() -> str:
    """Return the full CREATE TABLE + CREATE INDEX SQL."""
    return _SCHEMA
