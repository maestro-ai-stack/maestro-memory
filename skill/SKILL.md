---
name: maestro-memory-consolidate
description: "Consolidate files into long-term memory. Use when: user says 'consolidate memory', 'organize memory', 'import notes', or wants to batch-ingest files into maestro-memory. Triggers: consolidate, memory cleanup, import, reorganize memory, batch ingest."
---

# Memory Consolidation Skill

Batch-ingest files (markdown, notes, images) into maestro-memory with chunking and dedup.

## Steps

1. **Ask user for target paths** (if not specified). Default patterns:
   ```
   ~/.claude/projects/*/memory/*.md
   ~/progress/*/notes.md
   ```

2. **Dry run first** — always preview before writing:
   ```bash
   mmem consolidate --dry-run <paths...>
   ```
   Show the output to user: file count, chunk count, dedup skips.

3. **Confirm with user** — ask "Proceed with actual consolidation?" before writing.

4. **Run actual consolidation**:
   ```bash
   mmem consolidate <paths...>
   ```

5. **Show final stats** — files processed, facts added, entities created, errors.

## Notes

- Glob patterns are supported: `~/progress/**/*.md`
- Use `--project <name>` to scope memory to a specific project
- Image files (.png/.jpg/.jpeg) are OCR-extracted via GLM-OCR before ingestion
- PDF files are attempted as OCR first, then plain text fallback
- Dedup is automatic: identical chunks are skipped
