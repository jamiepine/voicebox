"""ModelScope download progress → the shared ``ProgressManager`` pipeline.

Mirrors ``hf_progress.create_hf_progress_callback`` for the HuggingFace path,
but ModelScope's SDK uses a different mechanism: a ``ProgressCallback``
class instantiated once per file (``callback_cls(filename, file_size)``)
with ``update(size)``/``end()`` methods, not a single function. Progress is
reported per-file (current/total bytes for *the file currently
downloading*), not aggregated across the whole snapshot — the SDK doesn't
expose a repo-wide total up front, and computing one would mean pre-listing
every file before download starts. Per-file progress still gives real
byte-level feedback, matching spec.md FR-012.
"""

from typing import Type


def create_modelscope_progress_callback_cls(model_name: str, progress_manager) -> Type:
    """Build a ``modelscope.hub.callback.ProgressCallback`` subclass that
    forwards updates to ``progress_manager`` under ``model_name``."""
    from modelscope.hub.callback import ProgressCallback

    class VoiceboxProgressCallback(ProgressCallback):
        def update(self, size: int) -> None:
            self._downloaded = getattr(self, "_downloaded", 0) + size
            progress_manager.update_progress(
                model_name=model_name,
                current=self._downloaded,
                total=self.file_size,
                filename=self.filename,
                status="downloading",
            )

    return VoiceboxProgressCallback
