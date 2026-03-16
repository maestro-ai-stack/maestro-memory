#!/bin/bash
# Stop hook: 对话结束时自动保存短期记忆到 maestro-memory
# 读取 stdin JSON 的 transcript_path，提取最后 ~2000 字符，写入 mmem
set -euo pipefail

# ── 安全退出：任何失败都不阻塞用户 ──────────────────────────────
trap 'exit 0' ERR

# ── 读取 stdin JSON ─────────────────────────────────────────────
INPUT="$(cat)"
TRANSCRIPT="$(echo "$INPUT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('transcript_path', ''))
" 2>/dev/null || true)"

[ -z "$TRANSCRIPT" ] && exit 0
[ ! -f "$TRANSCRIPT" ] && exit 0

# ── 提取最后若干条 assistant 消息（合计 ~2000 字符）──────────────
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
    # 取 content 文本
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

# 拼接并截断
text = '\n---\n'.join(msgs)[-2000:]
print(text)
" 2>/dev/null || true)"

[ -z "$TAIL" ] && exit 0

# ── 写入 mmem ──────────────────────────────────────────────────
mmem add --type conversation --source conversation "$TAIL" 2>/dev/null || true

exit 0
