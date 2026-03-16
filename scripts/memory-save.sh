#!/bin/bash
# Stop hook: auto-save short-term memory to maestro-memory on conversation end
# Read transcript_path from stdin JSON, extract last ~2000 chars, write to mmem
set -euo pipefail

# ── Safe exit: never block user on failure ────────────────────────
trap 'exit 0' ERR

# ── Read stdin JSON ───────────────────────────────────────────────
INPUT="$(cat)"
TRANSCRIPT="$(echo "$INPUT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('transcript_path', ''))
" 2>/dev/null || true)"

[ -z "$TRANSCRIPT" ] && exit 0
[ ! -f "$TRANSCRIPT" ] && exit 0

# ── Extract last N assistant messages (~2000 chars total) ────────
TAIL="$(python3 -c "
import json, sys

lines = open('$TRANSCRIPT', encoding='utf-8').readlines()
msgs = []
for line in reversed(lines):
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue
    if obj.get('role') != 'assistant':
        continue
    # Extract content text
    content = obj.get('content', '')
    if isinstance(content, list):
        content = ' '.join(
            b.get('text', '') for b in content if isinstance(b, dict)
        )
    if not content:
        continue
    msgs.insert(0, content)
    total = sum(len(m) for m in msgs)
    if total >= 2000:
        break

# Concatenate and truncate
text = '\n---\n'.join(msgs)[-2000:]
print(text)
" 2>/dev/null || true)"

[ -z "$TAIL" ] && exit 0

# ── Write to mmem ────────────────────────────────────────────────
mmem add --type conversation --source conversation "$TAIL" 2>/dev/null || true

exit 0
