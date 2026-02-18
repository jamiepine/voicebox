"""
Idle timer for auto-unloading models after inactivity.

Usage:
    timer = IdleTimer(timeout=180, on_timeout=backend.unload_model, label="TTS")
    timer.touch()  # Reset timer on each model use
    timer.cancel()  # Cancel before explicit unload
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


class IdleTimer:
    """Calls a callback after a period of inactivity.

    The timer lives on the asyncio event loop.  Call ``touch()`` every time
    the model is used to reset the countdown.  When the timeout elapses
    without a ``touch()``, ``on_timeout`` is invoked.
    """

    def __init__(
        self,
        timeout: float,
        on_timeout: callable,
        label: str = "model",
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.timeout = timeout
        self.on_timeout = on_timeout
        self.label = label
        self._loop = loop
        self._handle: asyncio.TimerHandle | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set (or update) the event loop used for scheduling."""
        self._loop = loop

    def touch(self):
        """Reset the idle countdown.  Safe to call from any context."""
        self.cancel()
        if self.timeout <= 0:
            return  # Disabled (serverless mode)
        if self._loop is None:
            return  # No loop yet — skip scheduling
        try:
            self._handle = self._loop.call_later(self.timeout, self._fire)
        except RuntimeError:
            # Loop is closed — nothing to schedule on
            pass

    def cancel(self):
        """Cancel the pending timeout (if any)."""
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

    def _fire(self):
        """Called when the timer expires."""
        self._handle = None
        logger.info(f"[IdleTimer] {self.label} idle for {self.timeout}s — unloading")
        try:
            self.on_timeout()
        except Exception:
            logger.exception(f"[IdleTimer] Error unloading {self.label}")
