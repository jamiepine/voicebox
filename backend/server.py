"""
Entry point for PyInstaller-bundled voicebox server.

This module provides an entry point that works with PyInstaller by using
absolute imports instead of relative imports.
"""

import sys
import logging
import multiprocessing

# Set up logging FIRST, before any imports that might fail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,  # Log to stderr so it's captured by Tauri
)
logger = logging.getLogger(__name__)

# Log startup immediately to confirm binary execution
logger.info("=" * 60)
logger.info("voicebox-server starting up...")
logger.info(f"Python version: {sys.version}")
logger.info(f"Executable: {sys.executable}")
logger.info(f"Arguments: {sys.argv}")
logger.info("=" * 60)

try:
    logger.info("Importing argparse...")
    import argparse
    logger.info("Importing uvicorn...")
    import uvicorn
    logger.info("Standard library imports successful")

    # Import the FastAPI app from the backend package
    logger.info("Importing backend.config...")
    from backend import config
    logger.info("Importing backend.database...")
    from backend import database
    logger.info("Importing backend.main (this may take a while due to torch/transformers)...")
    from backend.main import app
    logger.info("Backend imports successful")
except Exception as e:
    logger.error(f"Failed to import required modules: {e}", exc_info=True)
    sys.exit(1)

if __name__ == "__main__":
    # Required for PyInstaller + multiprocessing
    multiprocessing.freeze_support()
    
    # Force 'spawn' start method on macOS for consistency
    if sys.platform == 'darwin':
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    try:
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
        args, unknown = parser.parse_known_args()
        if unknown:
            logger.info(f"Ignoring unknown arguments: {unknown}")
        logger.info(f"Parsed arguments: host={args.host}, port={args.port}, data_dir={args.data_dir}")

        # Set data directory if provided
        if args.data_dir:
            logger.info(f"Setting data directory to: {args.data_dir}")
            config.set_data_dir(args.data_dir)

        # Initialize database after data directory is set
        logger.info("Initializing database...")
        database.init_db()
        logger.info("Database initialized successfully")

        logger.info(f"Starting uvicorn server on {args.host}:{args.port}...")
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
        )
    except Exception as e:
        logger.error(f"Server startup failed: {e}", exc_info=True)
        sys.exit(1)
