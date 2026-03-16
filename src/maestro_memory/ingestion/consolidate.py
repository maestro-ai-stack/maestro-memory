from __future__ import annotations

import glob as globlib
from dataclasses import dataclass, field
from pathlib import Path

from maestro_memory.core.memory import Memory
from maestro_memory.ingestion.chunker import chunk_text
from maestro_memory.ingestion.dedup import is_duplicate
from maestro_memory.ingestion.ocr import is_image, is_ocr_target, ocr_extract

# ── 结果统计 ────────────────────────────────────────────────────


@dataclass
class ConsolidateResult:
    files_processed: int = 0
    chunks_total: int = 0
    chunks_skipped: int = 0
    facts_added: int = 0
    entities_created: int = 0
    errors: list[str] = field(default_factory=list)


# ── Glob 展开 ──────────────────────────────────────────────────


def expand_paths(patterns: list[str]) -> list[Path]:
    """展开 glob 模式 + ~ 路径，去重保序"""
    seen: set[Path] = set()
    result: list[Path] = []
    for pat in patterns:
        expanded = str(Path(pat).expanduser())
        matches = sorted(globlib.glob(expanded, recursive=True))
        for m in matches:
            p = Path(m)
            if p.is_file() and p not in seen:
                seen.add(p)
                result.append(p)
    return result


# ── 文件读取（含 OCR）─────────────────────────────────────────


async def _read_file(path: Path) -> str:
    """读取文件内容：图片走 OCR，PDF 先 OCR 后 fallback 文本，其余直接读"""
    if is_image(path):
        return await ocr_extract(path)
    if path.suffix.lower() == ".pdf":
        text = await ocr_extract(path)
        if text:
            return text
        return path.read_text(encoding="utf-8", errors="replace")
    return path.read_text(encoding="utf-8")


# ── 主管线 ─────────────────────────────────────────────────────


async def consolidate(
    memory: Memory,
    paths: list[Path],
    *,
    source_type: str = "file",
    dry_run: bool = False,
) -> ConsolidateResult:
    """批量摄入文件：分块 → 去重 → 写入"""
    result = ConsolidateResult()
    ep = memory._embedding_provider  # noqa: SLF001

    for path in paths:
        try:
            text = await _read_file(path)
        except Exception as exc:
            result.errors.append(f"{path}: {exc}")
            continue
        if not text:
            continue

        result.files_processed += 1
        chunks = chunk_text(text)
        result.chunks_total += len(chunks)

        for chunk in chunks:
            dup, _ = await is_duplicate(chunk, memory.store, ep)
            if dup:
                result.chunks_skipped += 1
                continue

            if dry_run:
                result.facts_added += 1
                continue

            add_result = await memory.add(
                chunk,
                source_type=source_type,
                source_ref=str(path),
            )
            result.facts_added += add_result.facts_added
            result.entities_created += add_result.entities_created

    return result
