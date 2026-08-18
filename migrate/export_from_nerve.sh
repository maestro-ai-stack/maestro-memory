#!/usr/bin/env bash
# Export the mnerve (maestro-nerve) Postgres knowledge graph to JSONL.
#
# This is both the migration source and the first real backup of the store:
# until now the 773 facts existed only inside the `nerve-data` Docker volume,
# with no export path in the CLI.
#
# Embeddings are deliberately NOT exported. They are 1024-d Cloudflare vectors,
# present on only 539 of 773 nodes, and the target engine re-embeds with BGE-M3.
#
# Usage: migrate/export_from_nerve.sh [OUT_DIR]

set -euo pipefail

OUT_DIR="${1:-$HOME/.maestro/backups/nerve-export-$(date +%Y%m%d)}"
CONTAINER="${NERVE_PG_CONTAINER:-nerve-pg}"

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "error: container '$CONTAINER' not found" >&2
  exit 1
fi
if [[ -z "$(docker ps -q -f "name=^/${CONTAINER}$")" ]]; then
  echo "error: container '$CONTAINER' is not running; start it before exporting" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

psql_jsonl() {
  docker exec "$CONTAINER" psql -U postgres -d nerve -tA -c "$1"
}

echo "Exporting to $OUT_DIR"

# ── Facts (kg_nodes) ──────────────────────────────────────────
psql_jsonl "
SELECT row_to_json(t) FROM (
  SELECT id, node_type, name, properties, importance, confidence,
         layer, parent_id, source, created_by,
         created_at, updated_at, superseded_by,
         (embedding IS NOT NULL) AS had_embedding
  FROM kg_nodes ORDER BY id
) t;" > "$OUT_DIR/memories.jsonl"

# ── Edges (kg_edges), resolved to node names so ids need not survive ──
psql_jsonl "
SELECT row_to_json(t) FROM (
  SELECT e.id, e.edge_type, e.weight, e.confidence, e.discovered_by,
         e.metadata, e.created_at,
         a.node_type AS from_type, a.name AS from_name,
         b.node_type AS to_type,   b.name AS to_name
  FROM kg_edges e
  JOIN kg_nodes a ON a.id = e.from_node
  JOIN kg_nodes b ON b.id = e.to_node
  ORDER BY e.id
) t;" > "$OUT_DIR/edges.jsonl"

# ── Gold query set (serving_log) ──────────────────────────────
# `candidates` carries rank + per-channel attribution; `selected` is the label.
# Candidate node ids are resolved to (type, name) so the gold set is portable
# across engines — Postgres integer ids do not survive migration.
psql_jsonl "
SELECT row_to_json(t) FROM (
  SELECT s.query_id, s.query_text, s.session_id, s.created_at,
         s.latency_ms, s.source, s.channel_weights,
         s.selected, s.rejected,
         (
           SELECT json_agg(
             jsonb_set(
               c - 'name',
               '{ref}',
               COALESCE(to_jsonb(n.node_type || E'\t' || n.name), 'null'::jsonb)
             ) ORDER BY (c->>'rank')::int
           )
           FROM jsonb_array_elements(s.candidates) c
           LEFT JOIN kg_nodes n ON n.id = (c->>'node_id')::int
         ) AS candidates,
         (
           SELECT json_agg(n.node_type || E'\t' || n.name)
           FROM unnest(COALESCE(s.selected, ARRAY[]::int[])) sid
           JOIN kg_nodes n ON n.id = sid
         ) AS selected_refs
  FROM serving_log s
  WHERE s.candidates IS NOT NULL
  ORDER BY s.created_at
) t;" > "$OUT_DIR/gold_queries.jsonl"

# ── Manifest ──────────────────────────────────────────────────
count() { grep -c . "$1" 2>/dev/null || echo 0; }
M=$(count "$OUT_DIR/memories.jsonl")
E=$(count "$OUT_DIR/edges.jsonl")
G=$(count "$OUT_DIR/gold_queries.jsonl")

cat > "$OUT_DIR/manifest.json" <<EOF
{
  "exported_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "source": "maestro-nerve postgres (container ${CONTAINER}, db nerve)",
  "memories": ${M},
  "edges": ${E},
  "gold_queries": ${G},
  "embeddings_included": false,
  "note": "Edges and gold-query candidates reference nodes by 'type\\tname', not integer id, so the set is portable across engines."
}
EOF

echo "  memories.jsonl      ${M}"
echo "  edges.jsonl         ${E}"
echo "  gold_queries.jsonl  ${G}"
echo "Manifest: $OUT_DIR/manifest.json"
