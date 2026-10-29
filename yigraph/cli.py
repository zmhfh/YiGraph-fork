"""Installed entry point for workspace setup and the existing terminal engine."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="yigraph")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--home", type=Path, help="Workspace directory (overrides YIGRAPH_HOME)")
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("init", help="Create configuration, knowledge base and demo data")
    setup.add_argument("path", type=Path)
    commands.add_parser("demo", help="Run local graph analysis without a database or LLM")
    commands.add_parser("chat", help="Start the terminal AAG engine (requires engine extra)")
    args = parser.parse_args(argv)
    if args.home is not None:
        os.environ["YIGRAPH_HOME"] = str(args.home.expanduser().resolve())
    try:
        if args.command == "init":
            from .workspace import init_workspace

            print(init_workspace(args.path))
        elif args.command == "demo":
            from .demo import run_demo

            print(json.dumps(run_demo(), indent=2))
        else:
            from aag.utils.path_utils import DEFAULT_CONFIG_PATH

            if not DEFAULT_CONFIG_PATH.is_file():
                parser.error("Configuration missing. Run 'yigraph init <directory>', then use --home <directory>.")
            # Reuse the public lazy import so a missing extra has an actionable error.
            from yigraph import AAGEngine  # noqa: F401
            from aag.main import main as chat

            return asyncio.run(chat()) or 0
    except (OSError, ValueError, ModuleNotFoundError) as exc:
        parser.exit(1, f"yigraph: {exc}\n")
    return 0
