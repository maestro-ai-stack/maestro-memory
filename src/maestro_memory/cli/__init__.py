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
    port: int = typer.Option(19830, help="Server port"),
    project: str = typer.Option(None, help="Project name"),
    daemon: bool = typer.Option(True, "--daemon/--foreground", help="Run as background daemon"),
):
    """Start mmem daemon server (pre-loads models, holds DB open)."""
    import subprocess
    import sys
    from pathlib import Path

    if daemon:
        pid_file = Path.home() / ".maestro" / "memory" / "server.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "maestro_memory.server", "--port", str(port)]
        if project:
            cmd.extend(["--project", project])
        proc = subprocess.Popen(
            cmd,
            stdout=open(pid_file.with_suffix(".log"), "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pid_file.write_text(str(proc.pid))
        typer.echo(f"Server started (pid={proc.pid}, port={port})")
    else:
        import uvicorn

        from maestro_memory.server.app import create_app

        a = create_app(project=project)
        uvicorn.run(a, host="127.0.0.1", port=port)


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
