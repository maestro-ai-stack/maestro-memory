"""Daemon auto-management: ensure daemon is running before CLI operations."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx

DAEMON_URL = "http://localhost:19830"
PID_FILE = Path.home() / ".maestro" / "memory" / "server.pid"
LOG_FILE = Path.home() / ".maestro" / "memory" / "server.log"


def _is_daemon_running() -> bool:
    """Check if daemon responds to health check."""
    try:
        resp = httpx.get(f"{DAEMON_URL}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _start_daemon() -> bool:
    """Start daemon in background. Returns True if started successfully."""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "maestro_memory.server", "--port", "19830"],
        stdout=open(LOG_FILE, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))
    # Wait for warmup (models loading)
    for _ in range(30):  # max 30s
        time.sleep(1)
        if _is_daemon_running():
            return True
    return False


def ensure_daemon() -> bool:
    """Ensure daemon is running. Auto-start if not. Returns True if available."""
    if _is_daemon_running():
        return True
    return _start_daemon()
