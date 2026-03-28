from __future__ import annotations

import typer

from maestro_memory.cli.add import add_cmd
from maestro_memory.cli.config import config_app
from maestro_memory.cli.consolidate import consolidate_cmd
from maestro_memory.cli.graph import graph_cmd
from maestro_memory.cli.relate import relate_cmd
from maestro_memory.cli.search import search_cmd
from maestro_memory.cli.status import status_cmd

app = typer.Typer(name="mmem", help="maestro-memory: temporal hybrid memory for AI agents")

app.command("search")(search_cmd)
app.command("add")(add_cmd)
app.command("relate")(relate_cmd)
app.command("consolidate")(consolidate_cmd)
app.command("graph")(graph_cmd)
app.command("status")(status_cmd)
app.add_typer(config_app, name="config")


@app.command("server-start")
def server_start_cmd(
    foreground: bool = typer.Option(False, "--foreground", "-f", help="Run in foreground (not daemon)"),
    project: str = typer.Option(None, help="Project name"),
):
    """Start mmem daemon server (pre-loads models, holds DB open)."""
    if foreground:
        import uvicorn
        from maestro_memory.server.app import create_app
        app_ = create_app(project=project)
        uvicorn.run(app_, host="127.0.0.1", port=19830)
    else:
        from maestro_memory.cli.daemon import ensure_daemon
        if ensure_daemon():
            typer.echo("Server running on http://localhost:19830")
        else:
            typer.echo("Failed to start server. Check ~/.maestro/memory/server.log")


@app.command("server-install")
def server_install_cmd():
    """Install mmem daemon as macOS launchd service (auto-start on login)."""
    import platform
    import subprocess
    import sys
    from pathlib import Path

    if platform.system() != "Darwin":
        typer.echo("launchd is macOS only. Use systemd on Linux.")
        raise typer.Exit(1)

    python_path = sys.executable
    log_dir = Path.home() / ".maestro" / "memory"
    log_dir.mkdir(parents=True, exist_ok=True)

    plist_src = Path(__file__).parent.parent / "resources" / "com.maestro.memory.plist"
    plist_dst = Path.home() / "Library" / "LaunchAgents" / "com.maestro.memory.plist"

    content = plist_src.read_text()
    content = content.replace("__PYTHON_PATH__", python_path)
    content = content.replace("__LOG_DIR__", str(log_dir))
    content = content.replace("__HOME__", str(Path.home()))

    plist_dst.write_text(content)

    subprocess.run(["launchctl", "load", str(plist_dst)], check=True)
    typer.echo(f"Installed: {plist_dst}")
    typer.echo(f"Daemon will auto-start on login. Logs: {log_dir}/server.log")
    typer.echo("To check status: launchctl list | grep maestro")


@app.command("server-uninstall")
def server_uninstall_cmd():
    """Uninstall mmem daemon launchd service."""
    import subprocess
    from pathlib import Path

    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.maestro.memory.plist"
    if plist_path.exists():
        subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
        plist_path.unlink()
        typer.echo("Uninstalled mmem daemon service.")
    else:
        typer.echo("No launchd service found.")


@app.command("server-stop")
def server_stop_cmd():
    """Stop mmem daemon server."""
    import os
    import signal
    from pathlib import Path

    pid_file = Path.home() / ".maestro" / "memory" / "server.pid"
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            typer.echo(f"Server stopped (pid={pid})")
        except ProcessLookupError:
            typer.echo("Server not running")
        pid_file.unlink(missing_ok=True)
    else:
        typer.echo("No server pid file found")
