"""Tests for context string enrichment (Phase 0)."""

from __future__ import annotations

import pytest

from maestro_memory.ingestion.enrichment import enrich_for_embedding, enrich_template


class TestEnrichTemplate:
    """enrich_template — synchronous template-based enrichment."""

    def test_minimal_input(self):
        """No entity, no relations — still produces category + importance."""
        result = enrich_template("some fact")
        assert result.startswith("some fact\n")
        assert "Category: general observation." in result
        assert "Importance: NORMAL." in result
        # No entity or relation lines
        assert "Entity:" not in result
        assert "Related:" not in result

    def test_with_entity(self):
        result = enrich_template(
            "Prof X needs climate panel",
            entity_name="Prof X",
            entity_type="prospect",
        )
        assert "Entity: Prof X (prospect)." in result

    def test_with_related_entities(self):
        result = enrich_template(
            "ERA5 has gaps",
            related_entities=["CRU TS", "CHIRPS"],
        )
        assert "Related: CRU TS, CHIRPS." in result

    def test_with_relations(self):
        result = enrich_template(
            "Repurchase rate uses 30-day window",
            relations=[("affects", "revenue reporting"), ("derived_from", "raw transactions")],
        )
        assert "Affects: revenue reporting." in result
        assert "Derived from: raw transactions." in result

    def test_grouped_relations(self):
        """Multiple targets under the same predicate are grouped."""
        result = enrich_template(
            "fact",
            relations=[("causes", "A"), ("causes", "B")],
        )
        assert "Causes: A, B." in result

    def test_unknown_predicate_formatted(self):
        """Unknown predicates get capitalized with underscores replaced."""
        result = enrich_template(
            "fact",
            relations=[("my_custom_rel", "target")],
        )
        assert "My custom rel: target." in result

    def test_fact_type_mapping(self):
        for code, label in [
            ("observation", "general observation"),
            ("preference", "user preference"),
            ("feedback", "user feedback"),
            ("decision", "strategic decision"),
        ]:
            result = enrich_template("x", fact_type=code)
            assert f"Category: {label}." in result

    def test_unknown_fact_type_passthrough(self):
        result = enrich_template("x", fact_type="custom_type")
        assert "Category: custom_type." in result

    def test_importance_critical(self):
        result = enrich_template("x", importance=0.9)
        assert "Importance: CRITICAL." in result

    def test_importance_high(self):
        result = enrich_template("x", importance=0.7)
        assert "Importance: HIGH." in result

    def test_importance_normal(self):
        result = enrich_template("x", importance=0.5)
        assert "Importance: NORMAL." in result

    def test_importance_low(self):
        result = enrich_template("x", importance=0.2)
        assert "Importance: LOW." in result

    def test_importance_boundary_08(self):
        """0.8 is NOT > 0.8, so should be HIGH."""
        result = enrich_template("x", importance=0.8)
        assert "Importance: HIGH." in result

    def test_importance_boundary_06(self):
        result = enrich_template("x", importance=0.6)
        assert "Importance: NORMAL." in result

    def test_importance_boundary_03(self):
        result = enrich_template("x", importance=0.3)
        assert "Importance: LOW." in result

    def test_original_content_first_line(self):
        """Original content is always the first line."""
        content = "The original fact content"
        result = enrich_template(content, entity_name="E", importance=0.9)
        assert result.split("\n")[0] == content

    def test_full_enrichment(self):
        """All fields populated — smoke test the full output."""
        result = enrich_template(
            "Repurchase rate uses 30-day rolling window",
            fact_type="decision",
            entity_name="repurchase_metric",
            entity_type="metric",
            importance=0.85,
            related_entities=["is_test filter", "B2B segment"],
            relations=[("affects", "revenue reporting"), ("affects", "retention analysis")],
        )
        lines = result.split("\n")
        assert lines[0] == "Repurchase rate uses 30-day rolling window"
        assert "Category: strategic decision." in result
        assert "Entity: repurchase_metric (metric)." in result
        assert "Related: is_test filter, B2B segment." in result
        assert "Affects: revenue reporting, retention analysis." in result
        assert "Importance: CRITICAL." in result


class TestEnrichForEmbedding:
    """enrich_for_embedding — async wrapper."""

    @pytest.mark.asyncio
    async def test_async_matches_sync(self):
        """Async version produces identical output to sync template."""
        kwargs = dict(
            fact_type="feedback",
            entity_name="Claude",
            entity_type="tool",
            importance=0.7,
            related_entities=["mmem"],
            relations=[("uses", "SQLite")],
        )
        sync_result = enrich_template("test content", **kwargs)
        async_result = await enrich_for_embedding("test content", **kwargs)
        assert sync_result == async_result

    @pytest.mark.asyncio
    async def test_minimal(self):
        result = await enrich_for_embedding("bare fact")
        assert "bare fact" in result
        assert "Category:" in result
        assert "Importance:" in result
