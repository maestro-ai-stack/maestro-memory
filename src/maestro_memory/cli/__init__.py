from __future__ import annotations

import typer

from maestro_memory.cli.add import add_cmd
from maestro_memory.cli.config import config_app
from maestro_memory.cli.graph import graph_cmd
from maestro_memory.cli.search import search_cmd
from maestro_memory.cli.status import status_cmd

app = typer.Typer(name="mmem", help="maestro-memory: temporal hybrid memory for AI agents")

app.command("search")(search_cmd)
app.command("add")(add_cmd)
app.command("graph")(graph_cmd)
app.command("status")(status_cmd)
app.add_typer(config_app, name="config")
