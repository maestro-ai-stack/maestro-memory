from __future__ import annotations

from typer.testing import CliRunner

from maestro_memory.cli import mnerve


runner = CliRunner()


def test_understand_is_compact_and_emits_copyable_local_targets(monkeypatch) -> None:
    monkeypatch.setattr(
        mnerve,
        "_request",
        lambda *_args: {
            "results": [
                {
                    "fact_id": 42,
                    "content": "Ao asked whether Maestro can collaborate on the NUS Agentic AI grant.",
                    "entity_name": "Ao Wang",
                }
            ],
            "meta": {"confidence": "high"},
            "query_id": 17,
        },
    )

    result = runner.invoke(mnerve.app, ["understand", "Ao grant", "--limit", "3"])

    assert result.exit_code == 0, result.output
    assert "confidence=high" in result.output
    assert "fact:42 [Ao Wang]" in result.output
    assert "Recall [query:17]" in result.output
    assert "Feedback: mnerve feedback 'query:17'" in result.output
    assert "https://" not in result.output


def test_understand_hides_guidance_sentinel(monkeypatch) -> None:
    monkeypatch.setattr(
        mnerve,
        "_request",
        lambda *_args: {
            "results": [
                {"fact_id": -1, "content": "Key query terms not found"},
                {"fact_id": 42, "content": "Useful memory", "entity_name": None},
            ],
            "meta": {"confidence": "low"},
        },
    )

    result = runner.invoke(mnerve.app, ["understand", "Ao grant"])

    assert result.exit_code == 0, result.output
    assert "fact:-1" not in result.output
    assert "fact:42" in result.output


def test_feedback_decodes_query_and_records_fact_ids(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        mnerve,
        "_request",
        lambda method, path, payload=None: calls.append((method, path, payload)) or {"facts_updated": 1},
    )
    query_id = mnerve._query_id("Ao grant")

    result = runner.invoke(mnerve.app, ["feedback", query_id, "fact:42"])

    assert result.exit_code == 0, result.output
    assert calls == [("POST", "/feedback", {"query": "Ao grant", "used_fact_ids": [42]})]


def test_feedback_uses_short_server_query_id(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        mnerve,
        "_request",
        lambda method, path, payload=None: calls.append((method, path, payload)) or {"status": "ok", "facts_updated": 1},
    )

    result = runner.invoke(mnerve.app, ["feedback", "query:17", "fact:42"])

    assert result.exit_code == 0, result.output
    assert calls == [("POST", "/feedback", {"query_id": 17, "used_fact_ids": [42]})]


def test_remember_uses_local_add_contract(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        mnerve,
        "_request",
        lambda method, path, payload=None: calls.append((method, path, payload)) or {"episode_id": 7, "facts_added": 1},
    )

    result = runner.invoke(
        mnerve.app,
        ["remember", "Grant deadline is 28 Sep", "--type", "fact", "--idempotency-key", "gmail:1"],
    )

    assert result.exit_code == 0, result.output
    assert calls[0][0:2] == ("POST", "/add")
    assert calls[0][2]["fact_type"] == "observation"
    assert calls[0][2]["source_ref"] is None
    assert calls[0][2]["idempotency_key"] == "gmail:1"


def test_failure_has_no_fallback(monkeypatch) -> None:
    monkeypatch.setattr(mnerve, "_request", lambda *_args: (_ for _ in ()).throw(RuntimeError("daemon unavailable")))

    result = runner.invoke(mnerve.app, ["understand", "Ao grant"])

    assert result.exit_code == 1
    assert "Local daemon only; no fallback was attempted" in result.output
