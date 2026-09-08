# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-09-08

### Added
- Local-only `mnerve` compatibility CLI with compact recall and explicit feedback.
- Short server-issued `query:<id>` feedback tokens and retry-safe writes via
  `mnerve remember --idempotency-key`.
- Loopback daemon endpoint shared by the CLI, client, server, and launchd resource.
- Release gate covering lint, tests, package build, metadata, and local-only boundaries.

### Changed
- The base installation now includes everything required to run the local daemon.
- Public documentation now describes the shipped local SQLite architecture.

### Removed
- Container-backed legacy migration utilities and obsolete hosted-service plans.
- Implicit external LLM extraction triggered by ambient API-key environment variables.

## [0.2.0] - 2026-03-27

### Added
- **Cross-encoder reranking**: Lazy-loaded ms-marco-MiniLM-L-6-v2 reranker integrated into search pipeline. Graceful fallback when sentence-transformers not installed. (+20% QA accuracy on LongMemEval)
- **SessionState**: Implicit per-session context tracking — recent queries, accessed facts, entity activation, query expansion, session embedding (EMA). Auto-updated on every `search()` call.
- **LongMemEval benchmark**: Full evaluation infrastructure with retrieval eval and QA pipeline scripts.
- **Optuna parameter sweep**: Bayesian optimization for retrieval parameters.
- **Architecture v2 design**: Multi-tool MCP interface, 4-stage retrieval pipeline, continuous learning roadmap.

### Changed
- `search()` now accepts `rerank=True` parameter (default on). When cross-encoder is available, fetches 5x candidates for reranking.
- Built-in retrieval benchmark improved from 98% to 100%.

### Performance
- LongMemEval QA accuracy: 50% (baseline) -> 94% (v0.2.0)
- Surpassed Hindsight (91.4%), Zep (63.8%), Mem0 (49%)
- Key improvements: session decomposition (+16%), cross-encoder rerank (+20%)

## [0.1.0] - 2026-03-16

### Added
- Initial release: temporal hybrid memory system for AI agents.
- 4-signal retrieval: BM25 (FTS5) + embedding (sentence-transformers) + knowledge graph + ACT-R temporal decay.
- Reciprocal Rank Fusion (RRF) for multi-signal combination.
- LLM-based entity extraction (OpenAI / Anthropic) with no-LLM fallback.
- Per-project memory isolation via SHA256-hashed SQLite databases.
- CLI with search, add, graph, relate, consolidate, status, config commands.
- Python async SDK (`Memory` class).
- 3-tier deduplication (hash, cosine similarity, LLM placeholder).
- Chinese language support via jieba segmentation.
- 9-scenario retrieval benchmark suite.
