#!/usr/bin/env python3
"""
Test script to observe exactly how HuggingFace reports download progress
for each TTS model. Doesn't load models — just downloads and tracks tqdm.

Usage:
    backend/venv/bin/python scripts/test_download_progress.py qwen
    backend/venv/bin/python scripts/test_download_progress.py luxtts
    backend/venv/bin/python scripts/test_download_progress.py chatterbox

Add --delete to clear cache first and force a real download:
    backend/venv/bin/python scripts/test_download_progress.py chatterbox --delete
"""

import os
import shutil
import sys
import time
import threading
from pathlib import Path
from contextlib import contextmanager

# ─── Configuration ────────────────────────────────────────────────────────────

MODELS = {
    "qwen": {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "method": "from_pretrained",
        "description": "Qwen TTS 1.7B (uses transformers from_pretrained)",
    },
    "luxtts": {
        "repo_id": "YatharthS/LuxTTS",
        "method": "snapshot_download",
        "description": "LuxTTS (uses snapshot_download)",
    },
    "chatterbox": {
        "repo_id": "ResembleAI/chatterbox",
        "method": "snapshot_download",
        "allow_patterns": [
            "ve.pt",
            "t3_mtl23ls_v2.safetensors",
            "s3gen.pt",
            "grapheme_mtl_merged_expanded_v1.json",
            "conds.pt",
            "Cangjie5_TC.json",
        ],
        "description": "Chatterbox Multilingual (uses snapshot_download with allow_patterns)",
    },
}


# ─── Progress tracking (mirrors our HFProgressTracker) ────────────────────────

class ProgressSpy:
    """Intercepts tqdm to see exactly what HF reports."""

    def __init__(self):
        self._lock = threading.Lock()
        self.events = []  # List of dicts: {time, type, ...}
        self._original_tqdm_class = None
        self._original_tqdm_auto = None
        self._patched_modules = {}
        self._hf_tqdm_original_update = None
        self._start_time = None

    def _elapsed(self):
        return time.time() - self._start_time if self._start_time else 0

    def _log(self, event_type, **kwargs):
        entry = {"time": f"{self._elapsed():.1f}s", "type": event_type, **kwargs}
        self.events.append(entry)

        # Live print
        parts = [f"[{entry['time']:>7s}] {event_type:>10s}"]
        for k, v in kwargs.items():
            if k in ("current", "total") and isinstance(v, (int, float)) and v > 1_000_000:
                parts.append(f"{k}={v / 1_000_000:.1f}MB")
            else:
                parts.append(f"{k}={v}")
        print("  ".join(parts), flush=True)

    def _create_tracked_tqdm_class(self):
        spy = self
        original_tqdm = self._original_tqdm_class

        class SpyTqdm(original_tqdm):
            def __init__(self, *args, **kwargs):
                desc = kwargs.get("desc", "")
                if not desc and args:
                    first_arg = args[0]
                    if isinstance(first_arg, str):
                        desc = first_arg

                filename = ""
                if desc:
                    if ":" in desc:
                        filename = desc.split(":")[0].strip()
                    else:
                        filename = desc.strip()

                # Filter out non-standard kwargs
                tqdm_kwargs = {
                    'iterable', 'desc', 'total', 'leave', 'file', 'ncols',
                    'mininterval', 'maxinterval', 'miniters', 'ascii', 'disable',
                    'unit', 'unit_scale', 'dynamic_ncols', 'smoothing',
                    'bar_format', 'initial', 'position', 'postfix',
                    'unit_divisor', 'write_bytes', 'lock_args', 'nrows',
                    'colour', 'color', 'delay', 'gui', 'disable_default', 'pos',
                }
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in tqdm_kwargs}

                try:
                    super().__init__(*args, **filtered_kwargs)
                except TypeError:
                    super().__init__(*args, **kwargs)

                self._spy_filename = filename or "unknown"
                total = getattr(self, "total", None)

                spy._log(
                    "INIT",
                    filename=self._spy_filename,
                    total=total or 0,
                    unit=kwargs.get("unit", "?"),
                    unit_scale=kwargs.get("unit_scale", False),
                    disable=kwargs.get("disable", False),
                )

            def update(self, n=1):
                result = super().update(n)

                current = getattr(self, "n", 0)
                total = getattr(self, "total", 0)
                filename = self._spy_filename

                spy._log(
                    "UPDATE",
                    filename=filename,
                    n=n,
                    current=current,
                    total=total or 0,
                    pct=f"{100 * current / total:.1f}%" if total else "?",
                )

                return result

            def close(self):
                spy._log("CLOSE", filename=self._spy_filename)
                return super().close()

        return SpyTqdm

    @contextmanager
    def patch(self):
        """Context manager that patches tqdm globally — same as HFProgressTracker."""
        self._start_time = time.time()

        try:
            import tqdm as tqdm_module
            self._original_tqdm_class = tqdm_module.tqdm
        except ImportError:
            yield
            return

        tracked_tqdm = self._create_tracked_tqdm_class()

        # Patch tqdm.tqdm
        tqdm_module.tqdm = tracked_tqdm

        # Patch tqdm.auto.tqdm
        self._original_tqdm_auto = None
        if hasattr(tqdm_module, "auto") and hasattr(tqdm_module.auto, "tqdm"):
            self._original_tqdm_auto = tqdm_module.auto.tqdm
            tqdm_module.auto.tqdm = tracked_tqdm

        # Patch in sys.modules (same as HFProgressTracker)
        tqdm_attr_names = ['tqdm', 'base_tqdm', 'old_tqdm']
        patched_count = 0

        for module_name in list(sys.modules.keys()):
            if "huggingface" in module_name or module_name.startswith("tqdm"):
                try:
                    module = sys.modules[module_name]
                    for attr_name in tqdm_attr_names:
                        if hasattr(module, attr_name):
                            attr = getattr(module, attr_name)
                            is_tqdm_class = (
                                attr is self._original_tqdm_class
                                or (self._original_tqdm_auto and attr is self._original_tqdm_auto)
                                or (
                                    hasattr(attr, "__name__")
                                    and attr.__name__ == "tqdm"
                                    and hasattr(attr, "update")
                                )
                            )
                            if is_tqdm_class:
                                key = f"{module_name}.{attr_name}"
                                self._patched_modules[key] = (module, attr_name, attr)
                                setattr(module, attr_name, tracked_tqdm)
                                patched_count += 1
                except (AttributeError, TypeError):
                    pass

        # Monkey-patch HF's tqdm.update (same as HFProgressTracker)
        try:
            from huggingface_hub.utils import tqdm as hf_tqdm_module
            if hasattr(hf_tqdm_module, 'tqdm'):
                hf_tqdm_class = hf_tqdm_module.tqdm
                self._hf_tqdm_original_update = hf_tqdm_class.update
                spy = self

                def patched_update(tqdm_self, n=1):
                    result = spy._hf_tqdm_original_update(tqdm_self, n)
                    desc = getattr(tqdm_self, 'desc', '') or ''
                    current = getattr(tqdm_self, 'n', 0)
                    total = getattr(tqdm_self, 'total', 0) or 0

                    spy._log(
                        "HF_UPDATE",
                        desc=desc,
                        current=current,
                        total=total,
                        pct=f"{100 * current / total:.1f}%" if total else "?",
                    )
                    return result

                hf_tqdm_class.update = patched_update
                patched_count += 1
        except (ImportError, AttributeError):
            pass

        print(f"\n=== Patched {patched_count} tqdm references ===\n", flush=True)

        try:
            yield
        finally:
            # Restore everything
            import tqdm as tqdm_module
            tqdm_module.tqdm = self._original_tqdm_class
            if self._original_tqdm_auto:
                tqdm_module.auto.tqdm = self._original_tqdm_auto
            for key, (module, attr_name, original) in self._patched_modules.items():
                try:
                    setattr(module, attr_name, original)
                except (AttributeError, TypeError):
                    pass
            if self._hf_tqdm_original_update:
                try:
                    from huggingface_hub.utils import tqdm as hf_tqdm_module
                    if hasattr(hf_tqdm_module, 'tqdm'):
                        hf_tqdm_module.tqdm.update = self._hf_tqdm_original_update
                except (ImportError, AttributeError):
                    pass

    def summary(self):
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        inits = [e for e in self.events if e["type"] == "INIT"]
        updates = [e for e in self.events if e["type"] in ("UPDATE", "HF_UPDATE")]

        print(f"\ntqdm bars created: {len(inits)}")
        for e in inits:
            print(f"  - {e.get('filename', '?'):40s} total={e.get('total', '?')}")

        print(f"\nTotal update calls: {len(updates)}")

        # Group updates by filename
        by_file = {}
        for e in updates:
            fn = e.get("filename") or e.get("desc", "unknown")
            if fn not in by_file:
                by_file[fn] = []
            by_file[fn].append(e)

        for fn, evts in by_file.items():
            max_current = max(e.get("current", 0) for e in evts)
            max_total = max(e.get("total", 0) for e in evts)
            print(f"\n  {fn}:")
            print(f"    updates: {len(evts)}")
            print(f"    max current: {max_current:,}")
            print(f"    max total:   {max_total:,}")
            if max_total > 0 and max_current > 0:
                print(f"    final pct:   {100 * max_current / max_total:.1f}%")
            else:
                print(f"    final pct:   NO PROGRESS REPORTED")


# ─── Delete cache ─────────────────────────────────────────────────────────────

def delete_cache(repo_id: str):
    from huggingface_hub import constants as hf_constants
    cache_dir = Path(hf_constants.HF_HUB_CACHE)
    repo_cache = cache_dir / ("models--" + repo_id.replace("/", "--"))
    if repo_cache.exists():
        print(f"Deleting cache: {repo_cache}")
        shutil.rmtree(repo_cache)
        print("Deleted.")
    else:
        print(f"No cache found at {repo_cache}")


# ─── Download functions ───────────────────────────────────────────────────────

def download_qwen(spy: ProgressSpy):
    """Mirrors how pytorch_backend.py downloads Qwen."""
    from transformers import AutoModel
    repo_id = MODELS["qwen"]["repo_id"]

    print(f"Downloading {repo_id} via AutoModel.from_pretrained...")
    with spy.patch():
        # This is what Qwen3TTSModel.from_pretrained does under the hood
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id)


def download_luxtts(spy: ProgressSpy):
    """Mirrors how luxtts_backend.py downloads LuxTTS."""
    from huggingface_hub import snapshot_download
    repo_id = MODELS["luxtts"]["repo_id"]

    print(f"Downloading {repo_id} via snapshot_download...")
    with spy.patch():
        snapshot_download(repo_id)


def download_chatterbox(spy: ProgressSpy):
    """Mirrors how chatterbox_backend.py downloads Chatterbox."""
    from huggingface_hub import snapshot_download
    cfg = MODELS["chatterbox"]

    print(f"Downloading {cfg['repo_id']} via snapshot_download with allow_patterns...")
    with spy.patch():
        snapshot_download(
            repo_id=cfg["repo_id"],
            repo_type="model",
            revision="main",
            allow_patterns=cfg["allow_patterns"],
            token=os.getenv("HF_TOKEN"),
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MODELS:
        print(f"Usage: {sys.argv[0]} <{'|'.join(MODELS.keys())}> [--delete]")
        sys.exit(1)

    model_key = sys.argv[1]
    should_delete = "--delete" in sys.argv
    cfg = MODELS[model_key]

    print(f"\n{'=' * 70}")
    print(f"Testing download progress for: {cfg['description']}")
    print(f"Repo: {cfg['repo_id']}")
    print(f"Method: {cfg['method']}")
    print(f"{'=' * 70}\n")

    if should_delete:
        delete_cache(cfg["repo_id"])
        print()

    spy = ProgressSpy()

    dispatch = {
        "qwen": download_qwen,
        "luxtts": download_luxtts,
        "chatterbox": download_chatterbox,
    }

    try:
        dispatch[model_key](spy)
    except Exception as e:
        print(f"\n!!! Download failed: {e}")

    spy.summary()


if __name__ == "__main__":
    main()
