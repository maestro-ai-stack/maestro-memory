from __future__ import annotations

import re

# ── 分词器 ─────────────────────────────────────────────────────

_tokenizer = None


def _get_tokenizer():
    """尝试加载 tiktoken，失败则返回 None"""
    global _tokenizer  # noqa: PLW0603
    if _tokenizer is not None:
        return _tokenizer
    try:
        import tiktoken
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    except ImportError:
        _tokenizer = False  # 标记为不可用
    return _tokenizer


def _count_tokens(text: str) -> int:
    """估算 token 数：tiktoken 优先，fallback 到 word count * 1.33"""
    enc = _get_tokenizer()
    if enc:
        return len(enc.encode(text))
    return int(len(text.split()) * 1.33)


# ── 句子边界切分 ────────────────────────────────────────────────

_SENTENCE_RE = re.compile(r"(?<=\. )|(?<=\n)")


def _split_sentences(text: str) -> list[str]:
    """按句号或换行切分，保留分隔符在前一段"""
    parts = _SENTENCE_RE.split(text)
    return [p for p in parts if p]


# ── 主函数 ─────────────────────────────────────────────────────

def chunk_text(
    text: str,
    max_tokens: int = 256,
    overlap: float = 0.1,
) -> list[str]:
    """固定大小分块，尽量在句子边界切分，带 overlap"""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    overlap_tokens = int(max_tokens * overlap)
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for sent in sentences:
        sent_tokens = _count_tokens(sent)
        # 单句超限：直接作为独立 chunk
        if sent_tokens > max_tokens:
            if buf:
                chunks.append("".join(buf))
            chunks.append(sent.strip())
            buf, buf_tokens = [], 0
            continue

        if buf_tokens + sent_tokens > max_tokens:
            chunks.append("".join(buf))
            # overlap: 从末尾回溯
            tail, tail_tokens = [], 0
            for s in reversed(buf):
                st = _count_tokens(s)
                if tail_tokens + st > overlap_tokens:
                    break
                tail.insert(0, s)
                tail_tokens += st
            buf, buf_tokens = tail, tail_tokens

        buf.append(sent)
        buf_tokens += sent_tokens

    if buf:
        chunks.append("".join(buf))

    return [c.strip() for c in chunks if c.strip()]
