#!/usr/bin/env python3
"""VoxGuard entry point.

    python main.py

Reads configuration from .env (see .env.example), makes sure Voicebox is
reachable and Ollama is up before logging in, then starts the bot.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

from voxguard.bot import VoxGuardBot
from voxguard.config import Settings
from voxguard.runtime import Runtime


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # discord.py is chatty at INFO; keep it at WARNING unless debugging.
    if level != "DEBUG":
        logging.getLogger("discord").setLevel(logging.WARNING)
        logging.getLogger("discord.gateway").setLevel(logging.WARNING)


async def amain() -> None:
    settings = Settings.from_env()
    setup_logging(settings.log_level)
    log = logging.getLogger("voxguard")

    runtime = Runtime.build(settings)

    if not await runtime.voicebox.health():
        log.warning(
            "Voicebox isn't reachable at %s yet. Transcription and speech will fail "
            "until it's running. Start Voicebox and I'll pick it up automatically.",
            settings.voicebox_url,
        )

    # Start Ollama and pull the model up front. Doing it here rather than
    # lazily on first use means the multi-gigabyte download happens while the
    # operator is watching the console, not silently during someone's first
    # /chat — which otherwise just looks like the bot hanging.
    try:
        await runtime.ollama.ensure_server()
        log.info("Ollama reachable at %s", settings.ollama_host)
        log.info("Ensuring model '%s' is available...", settings.ollama_model)
        await runtime.ollama.ensure_model(settings.ollama_model)
        log.info("Model '%s' ready.", settings.ollama_model)
    except Exception as exc:
        log.warning("%s", exc)
        log.warning(
            "AI features (/chat, /talk-ai, /vctalk, /roam, /aimod) stay offline until "
            "Ollama is reachable. Everything else works."
        )

    bot = VoxGuardBot(settings, runtime)

    try:
        async with bot:
            await bot.start(settings.discord_token)
    finally:
        await runtime.aclose()


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
