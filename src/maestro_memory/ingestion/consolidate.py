from __future__ import annotations

import glob as globlib
from dataclasses import dataclass, field
from pathlib import Path

from maestro_memory.core.memory import Memory
from maestro_memory.ingestion.chunker import chunk_text
from maestro_memory.ingestion.dedup import is_duplicate
from maestro_memory.ingestion.ocr import is_image, is_ocr_target, ocr_extract

# ── Result stats ────────────────────────────────────────────────────


@dataclass
class ConsolidateResult:
    files_processed: int = 0
    chunks_total: int = 0
    chunks_skipped: int = 0
    facts_added: int = 0
    entities_created: int = 0
    errors: list[str] = field(default_factory=list)


# ── Glob expansion ──────────────────────────────────────────────────


def expand_paths(patterns: list[str]) -> list[Path]:
    """Expand glob patterns + ~ paths, deduplicate while preserving order."""
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


# ── File reading (with OCR) ─────────────────────────────────────────


async def _read_file(path: Path) -> str:
    """Read file content: images via OCR, PDFs try OCR then fallback to text, others read directly."""
    if is_image(path):
        return await ocr_extract(path)
    if path.suffix.lower() == ".pdf":
        text = await ocr_extract(path)
        if text:
            return text
        return path.read_text(encoding="utf-8", errors="replace")
    return path.read_text(encoding="utf-8")


# ── Main pipeline ─────────────────────────────────────────────────────


async def consolidate(
    memory: Memory,
    paths: list[Path],
    *,
    source_type: str = "file",
    dry_run: bool = False,
) -> ConsolidateResult:
    """Batch-ingest files: chunk -> dedup -> write."""
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
