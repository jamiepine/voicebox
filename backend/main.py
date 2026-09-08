"""Entry point for the voicebox backend.

Imports the configured FastAPI app and provides a ``python -m backend.main``
entry point for development.
"""

import argparse

import uvicorn

from . import config, database
from .app import app  # noqa: F401 -- re-export for uvicorn "backend.main:app"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="voicebox backend server")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (use 0.0.0.0 for remote access)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory for database, profiles, and generated audio",
    )
    args = parser.parse_args()

    if args.data_dir:
        config.set_data_dir(args.data_dir)

    database.init_db()

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=False,
    )
