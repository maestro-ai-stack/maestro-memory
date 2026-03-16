from __future__ import annotations

import re

# ── Tokenizer ─────────────────────────────────────────────────────

_tokenizer = None


def _get_tokenizer():
    """Try loading tiktoken, return None on failure."""
    global _tokenizer  # noqa: PLW0603
    if _tokenizer is not None:
        return _tokenizer
    try:
        import tiktoken
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    except ImportError:
        _tokenizer = False  # Mark as unavailable
    return _tokenizer


def _count_tokens(text: str) -> int:
    """Estimate token count: tiktoken preferred, fallback to word count * 1.33."""
    enc = _get_tokenizer()
    if enc:
        return len(enc.encode(text))
    return int(len(text.split()) * 1.33)


# ── Sentence boundary splitting ────────────────────────────────────────────────

_SENTENCE_RE = re.compile(r"(?<=\. )|(?<=\n)")


def _split_sentences(text: str) -> list[str]:
    """Split by period or newline, keeping delimiter with preceding segment."""
    parts = _SENTENCE_RE.split(text)
    return [p for p in parts if p]


# ── Main function ─────────────────────────────────────────────────────

def chunk_text(
    text: str,
    max_tokens: int = 256,
    overlap: float = 0.1,
) -> list[str]:
    """Fixed-size chunking at sentence boundaries with overlap."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    overlap_tokens = int(max_tokens * overlap)
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for sent in sentences:
        sent_tokens = _count_tokens(sent)
        # Single sentence exceeds limit: emit as standalone chunk
        if sent_tokens > max_tokens:
            if buf:
                chunks.append("".join(buf))
            chunks.append(sent.strip())
            buf, buf_tokens = [], 0
            continue

        if buf_tokens + sent_tokens > max_tokens:
            chunks.append("".join(buf))
            # overlap: backtrack from end
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
