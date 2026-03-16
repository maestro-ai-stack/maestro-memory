from __future__ import annotations

import typer

from maestro_memory.core.config import CONFIG_PATH, load_config, write_default_config

config_app = typer.Typer(help="Manage maestro-memory configuration")


@config_app.command("init")
def config_init() -> None:
    """Generate default config.toml."""
    if CONFIG_PATH.exists():
        typer.echo(f"Config already exists at {CONFIG_PATH}")
        overwrite = typer.confirm("Overwrite?", default=False)
        if not overwrite:
            raise typer.Abort()
    path = write_default_config()
    typer.echo(f"Config written to {path}")


@config_app.command("show")
def config_show() -> None:
    """Show current configuration."""
    cfg = load_config()
    for section, values in cfg.items():
        typer.echo(f"[{section}]")
        if isinstance(values, dict):
            for k, v in values.items():
                typer.echo(f"  {k} = {v}")
        else:
            typer.echo(f"  {values}")
