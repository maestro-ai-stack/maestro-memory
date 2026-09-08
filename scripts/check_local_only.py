#!/usr/bin/env python3
"""Fail the release if a hosted or container runtime path returns."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOTS = (ROOT / "src", ROOT / "skills")
FORBIDDEN = (
    "nerve-api.maestro.onl",
    "workers.dev",
    "cloudflare",
    "fly.io",
    "railway",
    "docker",
    "openai",
    "anthropic",
)
EXPECTED_ENDPOINT = 'DAEMON_HOST = "127.0.0.1"\nDAEMON_PORT = 19830'


def main() -> None:
    violations: list[str] = []
    for root in RUNTIME_ROOTS:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix in {".pyc", ".png"}:
                continue
            text = path.read_text(errors="ignore").lower()
            for marker in FORBIDDEN:
                if marker in text:
                    violations.append(f"{path.relative_to(ROOT)}: {marker}")

    if violations:
        joined = "\n".join(violations)
        raise SystemExit(f"non-local runtime references found:\n{joined}")

    endpoint_source = (ROOT / "src/maestro_memory/server/config.py").read_text()
    if EXPECTED_ENDPOINT not in endpoint_source:
        raise SystemExit("local daemon endpoint authority is missing or changed")

    print("local-only boundary: PASS")


if __name__ == "__main__":
    main()
