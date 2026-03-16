from __future__ import annotations

import hashlib
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 fallback (shouldn't happen with >=3.11)
    import tomli as tomllib  # type: ignore[no-redef]

BASE_DIR = Path.home() / ".maestro" / "memory"
CONFIG_PATH = BASE_DIR / "config.toml"

DEFAULT_CONFIG: dict = {
    "embedding": {"provider": "local", "model": "all-MiniLM-L6-v2"},
    "llm": {"provider": "none"},
    "storage": {"base_dir": str(BASE_DIR)},
}


def load_config() -> dict:
    """Load config from TOML file, falling back to defaults."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "rb") as f:
            user_cfg = tomllib.load(f)
        merged = {**DEFAULT_CONFIG}
        for section, values in user_cfg.items():
            if isinstance(values, dict) and section in merged:
                merged[section] = {**merged[section], **values}
            else:
                merged[section] = values
        return merged
    return dict(DEFAULT_CONFIG)


def write_default_config() -> Path:
    """Write default config.toml and return its path."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '[embedding]',
        'provider = "local"',
        'model = "all-MiniLM-L6-v2"',
        '',
        '[llm]',
        'provider = "none"  # "openai" | "anthropic" | "none"',
        '',
        '[storage]',
        f'base_dir = "{BASE_DIR}"',
        '',
    ]
    CONFIG_PATH.write_text("\n".join(lines))
    return CONFIG_PATH


def get_db_path(project: str | None = None) -> Path:
    """Return the database path for a project (or default)."""
    cfg = load_config()
    base = Path(cfg["storage"]["base_dir"])
    if project:
        project_hash = hashlib.sha256(project.encode()).hexdigest()[:12]
        db_dir = base / project_hash
    else:
        db_dir = base / "default"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "mem.db"
