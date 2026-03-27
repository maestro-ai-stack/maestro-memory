"""Run mmem daemon: python -m maestro_memory.server [--port 19830]."""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="maestro-memory daemon server")
    parser.add_argument("--port", type=int, default=19830)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    import uvicorn

    from maestro_memory.server.app import create_app

    app = create_app(project=args.project)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
