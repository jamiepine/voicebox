"""
HuggingFace Hub download progress tracking.
"""

import logging
from typing import Optional, Callable
from contextlib import contextmanager
import threading
import sys

logger = logging.getLogger(__name__)


class HFProgressTracker:
    """Tracks HuggingFace Hub download progress by intercepting tqdm."""
    
    def __init__(self, progress_callback: Optional[Callable] = None, filter_non_downloads: bool = False):
        self.progress_callback = progress_callback
        self.filter_non_downloads = filter_non_downloads  # Only filter if True
        self._original_tqdm_class = None
        self._lock = threading.Lock()
        self._total_downloaded = 0
        self._total_size = 0
        self._file_sizes = {}  # Track sizes of individual files
        self._file_downloaded = {}  # Track downloaded bytes per file
        self._current_filename = ""
        self._active_tqdms = {}  # Track active tqdm instances
        self._hf_tqdm_original_update = None  # For monkey-patching hf's tqdm
    
    def _create_tracked_tqdm_class(self):
        """Create a tqdm subclass that tracks progress."""
        tracker = self
        original_tqdm = self._original_tqdm_class
        
        class TrackedTqdm(original_tqdm):
            """A tqdm subclass that reports progress to our tracker."""

            def __init__(self, *args, **kwargs):
                # Extract filename from desc before passing to parent
                desc = kwargs.get("desc", "")
                if not desc and args:
                    first_arg = args[0]
                    if isinstance(first_arg, str):
                        desc = first_arg

                filename = ""
                if desc:
                    # Try to extract filename from description
                    # HuggingFace Hub uses format like "model.safetensors: 0%|..."
                    if ":" in desc:
                        filename = desc.split(":")[0].strip()
                    else:
                        filename = desc.strip()

                # When model is cached, suppress all tqdm output (e.g. "Fetching 12 files")
                # to keep the console clean on repeat startups.
                if tracker.filter_non_downloads:
                    kwargs['disable'] = True

                # Filter out non-standard kwargs that huggingface_hub might pass
                # These are custom kwargs that tqdm doesn't understand
                filtered_kwargs = {}
                # Known tqdm kwargs - pass these through
                tqdm_kwargs = {
                    'iterable', 'desc', 'total', 'leave', 'file', 'ncols', 'mininterval',
                    'maxinterval', 'miniters', 'ascii', 'disable', 'unit', 'unit_scale',
                    'dynamic_ncols', 'smoothing', 'bar_format', 'initial', 'position',
                    'postfix', 'unit_divisor', 'write_bytes', 'lock_args', 'nrows',
                    'colour', 'color', 'delay', 'gui', 'disable_default', 'pos'
                }
                for key, value in kwargs.items():
                    if key in tqdm_kwargs:
                        filtered_kwargs[key] = value

                # Try to initialize with filtered kwargs, fall back to all kwargs if that fails
                try:
                    super().__init__(*args, **filtered_kwargs)
                except TypeError:
                    # If filtering failed, try with all kwargs (maybe tqdm version accepts them)
                    super().__init__(*args, **kwargs)
                
                self._tracker_filename = filename or "unknown"
                
                with tracker._lock:
                    if filename:
                        tracker._current_filename = filename
                    tracker._active_tqdms[id(self)] = {
                        "filename": self._tracker_filename,
                    }
            
            def update(self, n=1):
                result = super().update(n)

                # Report progress
                with tracker._lock:
                    if id(self) in tracker._active_tqdms:
                        filename = tracker._active_tqdms[id(self)]["filename"]
                        current = getattr(self, "n", 0)
                        total = getattr(self, "total", 0)
                        
                        if total and total > 0:
                            # Always filter out non-byte progress bars (e.g., "Fetching 12 files")
                            # These cause crazy percentages because they're counting files, not bytes
                            if self._is_non_byte_progress(filename):
                                return result
                            
                            # When model is cached, also filter out generation-related progress
                            if tracker.filter_non_downloads:
                                if not self._is_download_progress(filename):
                                    return result
                            
                            # Update per-file tracking
                            tracker._file_sizes[filename] = total
                            tracker._file_downloaded[filename] = current
                            
                            # Calculate totals across all files
                            tracker._total_size = sum(tracker._file_sizes.values())
                            tracker._total_downloaded = sum(tracker._file_downloaded.values())
                            
                            # Only report progress once we have a meaningful total (at least 1MB)
                            # This avoids the "100% at 0MB" issue when small config
                            # files are counted before the real model files
                            MIN_TOTAL_BYTES = 1_000_000  # 1MB
                            if tracker._total_size < MIN_TOTAL_BYTES:
                                return result
                            
                            # Call progress callback
                            if tracker.progress_callback:
                                tracker.progress_callback(
                                    tracker._total_downloaded,
                                    tracker._total_size,
                                    filename
                                )
                
                return result
            
            def _is_non_byte_progress(self, filename: str) -> bool:
                """Check if this progress bar should be SKIPPED (returns True to skip).
                
                We want to track byte-based progress bars. This method identifies
                progress bars that count files/items instead of bytes, which would
                cause crazy percentages if mixed with our byte counting.
                
                Returns:
                    True = SKIP this bar (it's not byte-based)
                    False = TRACK this bar (it counts bytes)
                """
                if not filename:
                    return False
                
                filename_lower = filename.lower()
                
                # Skip "Fetching X files" - it counts files (total=12), not bytes
                # Don't skip "Downloading (incomplete total...)" - that IS byte-based
                skip_patterns = [
                    'fetching',  # "Fetching 12 files" has total=12 files, not bytes
                ]
                return any(pattern in filename_lower for pattern in skip_patterns)
            
            def _is_download_progress(self, filename: str) -> bool:
                """Check if this is a real file download progress bar vs internal processing."""
                if not filename or filename == "unknown":
                    return False
                
                # Real downloads have file extensions
                download_extensions = [
                    '.safetensors', '.bin', '.pt', '.pth',  # Model weights
                    '.json', '.txt', '.py',  # Config files
                    '.msgpack', '.h5',  # Other formats
                ]
                
                filename_lower = filename.lower()
                has_extension = any(filename_lower.endswith(ext) for ext in download_extensions)
                
                # Skip generation-related progress indicators
                skip_patterns = ['segment', 'processing', 'generating', 'loading']
                has_skip_pattern = any(pattern in filename_lower for pattern in skip_patterns)
                
                return has_extension and not has_skip_pattern
            
            def close(self):
                with tracker._lock:
                    if id(self) in tracker._active_tqdms:
                        del tracker._active_tqdms[id(self)]
                return super().close()
        
        return TrackedTqdm
    
    @contextmanager
    def patch_download(self):
        """Context manager to patch tqdm for progress tracking."""
        try:
            import tqdm as tqdm_module

            # Store original tqdm class
            self._original_tqdm_class = tqdm_module.tqdm
            
            # Reset totals
            with self._lock:
                self._total_downloaded = 0
                self._total_size = 0
                self._file_sizes = {}
                self._file_downloaded = {}
                self._current_filename = ""
                self._active_tqdms = {}
            
            # Create our tracked tqdm class
            tracked_tqdm = self._create_tracked_tqdm_class()

            # Patch tqdm.tqdm
            tqdm_module.tqdm = tracked_tqdm

            # Also patch tqdm.auto.tqdm if it exists (used by huggingface_hub)
            self._original_tqdm_auto = None
            if hasattr(tqdm_module, "auto") and hasattr(tqdm_module.auto, "tqdm"):
                self._original_tqdm_auto = tqdm_module.auto.tqdm
                tqdm_module.auto.tqdm = tracked_tqdm
            
            # NOTE: We intentionally do NOT replace tqdm classes in sys.modules
            # with TrackedTqdm. That approach causes `super()` errors because
            # huggingface_hub's tqdm subclass instances aren't TrackedTqdm instances.
            # Instead, we monkey-patch .update() directly on HF's tqdm class below.
            self._patched_modules = {}
            patched_count = 0

            # Monkey-patch the update method on huggingface_hub's tqdm class.
            # `from huggingface_hub.utils import tqdm` imports the CLASS directly
            # (not a module), so we patch .update on the class itself.
            self._hf_tqdm_original_update = None
            try:
                from huggingface_hub.utils import tqdm as hf_tqdm_class
                self._hf_tqdm_original_update = hf_tqdm_class.update

                # Create a wrapper that calls our tracking
                tracker = self  # Reference to HFProgressTracker instance
                def patched_update(tqdm_self, n=1):
                    result = tracker._hf_tqdm_original_update(tqdm_self, n)

                    # Track this progress
                    with tracker._lock:
                        desc = getattr(tqdm_self, 'desc', '') or ''
                        current = getattr(tqdm_self, 'n', 0)
                        total = getattr(tqdm_self, 'total', 0) or 0

                        # Skip non-byte progress bars
                        if 'fetching' in desc.lower():
                            return result

                        # Skip until we have a meaningful total (at least 1MB)
                        MIN_TOTAL_BYTES = 1_000_000  # 1MB
                        if total >= MIN_TOTAL_BYTES:
                            tracker._total_downloaded = current
                            tracker._total_size = total

                            if tracker.progress_callback:
                                tracker.progress_callback(current, total, desc)

                    return result

                hf_tqdm_class.update = patched_update
                patched_count += 1
            except (ImportError, AttributeError):
                pass

            if not self.filter_non_downloads:
                logger.debug(f"[HFProgressTracker] Patched {patched_count} tqdm references")
            
            yield
            
        except ImportError:
            # If tqdm not available, just yield without patching
            yield
        finally:
            # Restore original tqdm
            if self._original_tqdm_class:
                try:
                    import tqdm as tqdm_module
                    tqdm_module.tqdm = self._original_tqdm_class
                    
                    if self._original_tqdm_auto:
                        tqdm_module.auto.tqdm = self._original_tqdm_auto
                    
                    # Restore patched modules
                    for key, (module, attr_name, original) in self._patched_modules.items():
                        try:
                            if module and original:
                                setattr(module, attr_name, original)
                        except (AttributeError, TypeError):
                            pass
                    self._patched_modules = {}
                    
                    # Restore hf_tqdm's original update method
                    if self._hf_tqdm_original_update:
                        try:
                            from huggingface_hub.utils import tqdm as hf_tqdm_class
                            hf_tqdm_class.update = self._hf_tqdm_original_update
                        except (ImportError, AttributeError):
                            pass
                        self._hf_tqdm_original_update = None
                    
                except (ImportError, AttributeError):
                    pass


@contextmanager
def hf_offline_for_cached(is_cached: bool):
    """Force HF offline mode when model is cached to skip remote validation ('Fetching N files').

    huggingface_hub caches HF_HUB_OFFLINE at import time in constants.py,
    so setting the env var after import has no effect. We must patch the
    module-level constant directly.
    """
    if not is_cached:
        yield
        return

    import os
    from huggingface_hub import constants as hf_constants

    old_env = os.environ.get("HF_HUB_OFFLINE")
    old_const = hf_constants.HF_HUB_OFFLINE

    os.environ["HF_HUB_OFFLINE"] = "1"
    hf_constants.HF_HUB_OFFLINE = True

    try:
        yield
    finally:
        if old_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_env
        hf_constants.HF_HUB_OFFLINE = old_const


def create_hf_progress_callback(model_name: str, progress_manager):
    """Create a progress callback for HuggingFace downloads."""
    def callback(downloaded: int, total: int, filename: str = ""):
        """Progress callback.
        
        Note: We send updates even when total=0 (unknown) to provide feedback
        during the "incomplete total" phase of huggingface_hub downloads.
        The frontend handles total=0 gracefully.
        """
        progress_manager.update_progress(
            model_name=model_name,
            current=downloaded,
            total=total,
            filename=filename or "",
            status="downloading",
        )
    return callback
