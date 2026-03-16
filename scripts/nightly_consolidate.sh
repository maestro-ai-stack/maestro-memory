#!/bin/bash
# Nightly memory consolidation — like human sleep memory processing.
# Scans recent Claude Code sessions, extracts key facts, stores in mmem.
#
# Install as cron:
#   crontab -e
#   0 3 * * * /Users/ding/maestro/projects/maestro-memory/scripts/nightly_consolidate.sh >> /tmp/mmem-nightly.log 2>&1
#
# What it does:
# 1. Find session files modified in the last 24 hours
# 2. Extract user messages from each session
# 3. Feed to LLM for fact extraction (if API key available)
# 4. Store extracted facts in mmem
# 5. Log summary

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SESSIONS_DIR="$HOME/.claude/projects"
VENV="$SCRIPT_DIR/../.venv/bin/python"
MMEM="$HOME/.local/bin/mmem"

# Fallback to global mmem if venv not available
if [ ! -f "$VENV" ]; then
    VENV="python3"
fi

if ! command -v mmem &>/dev/null; then
    echo "[$(date)] mmem not found on PATH, skipping consolidation"
    exit 0
fi

echo "[$(date)] === Nightly memory consolidation starting ==="

# Find sessions modified in last 24 hours (skip subagents)
RECENT_SESSIONS=$(find "$SESSIONS_DIR" -maxdepth 2 -name "*.jsonl" \
    -not -path "*/subagents/*" \
    -mtime -1 \
    2>/dev/null || true)

COUNT=$(echo "$RECENT_SESSIONS" | grep -c "\.jsonl$" || echo 0)
echo "[$(date)] Found $COUNT sessions modified in last 24h"

if [ "$COUNT" -eq 0 ]; then
    echo "[$(date)] No recent sessions, nothing to consolidate"
    exit 0
fi

# Extract user messages and feed to mmem
echo "$RECENT_SESSIONS" | while read -r session_file; do
    [ -z "$session_file" ] && continue

    # Extract project name from path
    PROJECT=$(basename "$(dirname "$session_file")")
    SESSION_ID=$(basename "$session_file" .jsonl)

    echo "[$(date)] Processing: $PROJECT / $SESSION_ID"

    # Extract user messages (skip tool results, system messages, commands)
    USER_MSGS=$($VENV -c "
import json, sys
msgs = []
for line in open('$session_file'):
    try:
        d = json.loads(line)
        if d.get('type') != 'user':
            continue
        msg = d.get('message', {})
        content = msg.get('content', '')
        if isinstance(content, str) and len(content) > 20:
            # Skip tool results, system reminders, command outputs
            if content.startswith('<') or 'tool_result' in content:
                continue
            msgs.append(content[:500])
    except:
        pass
# Output condensed summary
print('\n---\n'.join(msgs[:20]))  # Max 20 messages per session
" 2>/dev/null || echo "")

    if [ -z "$USER_MSGS" ] || [ ${#USER_MSGS} -lt 30 ]; then
        continue
    fi

    # Store as episode with context
    mmem add "Session $SESSION_ID ($PROJECT): $(echo "$USER_MSGS" | head -c 1000)" \
        --source "nightly-consolidation" \
        --type observation \
        2>/dev/null || true

done

# Print final stats
echo "[$(date)] === Consolidation complete ==="
mmem status 2>/dev/null || true
