"""
Shared utilities for TTS/STT backend implementations.

Eliminates duplication of cache checking, device detection,
voice prompt combination, and model loading progress tracking.
"""

import logging
import platform
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..utils.audio import normalize_audio, load_audio
from ..utils.progress import get_progress_manager
from ..utils.hf_progress import HFProgressTracker, create_hf_progress_callback
from ..utils.tasks import get_task_manager

logger = logging.getLogger(__name__)


def is_model_cached(
    hf_repo: str,
    *,
    weight_extensions: tuple[str, ...] = (".safetensors", ".bin"),
    required_files: Optional[list[str]] = None,
) -> bool:
    """
    Check if a HuggingFace model is fully cached locally.

    Args:
        hf_repo: HuggingFace repo ID (e.g. "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        weight_extensions: File extensions that count as model weights.
        required_files: If set, check that these specific filenames exist
                        in snapshots instead of checking by extension.

    Returns:
        True if model is fully cached, False if missing or incomplete.
    """
    try:
        from huggingface_hub import constants as hf_constants

        repo_cache = Path(hf_constants.HF_HUB_CACHE) / ("models--" + hf_repo.replace("/", "--"))

        if not repo_cache.exists():
            return False

        # Incomplete blobs mean a download is still in progress
        blobs_dir = repo_cache / "blobs"
        if blobs_dir.exists() and any(blobs_dir.glob("*.incomplete")):
            logger.debug(f"Found .incomplete files for {hf_repo}")
            return False

        snapshots_dir = repo_cache / "snapshots"
        if not snapshots_dir.exists():
            return False

        if required_files:
            # Check that every required filename exists somewhere in snapshots
            for fname in required_files:
                if not any(snapshots_dir.rglob(fname)):
                    return False
            return True

        # Check that at least one weight file exists
        for ext in weight_extensions:
            if any(snapshots_dir.rglob(f"*{ext}")):
                return True

        logger.debug(f"No model weights found for {hf_repo}")
        return False

    except Exception as e:
        logger.warning(f"Error checking cache for {hf_repo}: {e}")
        return False


def get_torch_device(
    *,
    allow_xpu: bool = False,
    allow_directml: bool = False,
    allow_mps: bool = False,
    force_cpu_on_mac: bool = False,
) -> str:
    """
    Detect the best available torch device.

    Args:
        allow_xpu: Check for Intel XPU (IPEX) support.
        allow_directml: Check for DirectML (Windows) support.
        allow_mps: Allow MPS (Apple Silicon). If False, MPS falls back to CPU.
        force_cpu_on_mac: Force CPU on macOS regardless of GPU availability.
    """
    if force_cpu_on_mac and platform.system() == "Darwin":
        return "cpu"

    import torch

    if torch.cuda.is_available():
        return "cuda"

    if allow_xpu:
        try:
            import intel_extension_for_pytorch  # noqa: F401

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return "xpu"
        except ImportError:
            pass

    if allow_directml:
        try:
            import torch_directml

            if torch_directml.device_count() > 0:
                return torch_directml.device(0)
        except ImportError:
            pass

    if allow_mps:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

    return "cpu"


def check_cuda_compatibility() -> tuple[bool, str | None]:
    """Check if the installed PyTorch supports the current GPU's compute capability.

    Returns:
        (compatible, warning_message) — compatible is True if OK or no CUDA GPU,
        warning_message is a human-readable string if there's a problem.
    """
    import torch

    if not torch.cuda.is_available():
        return True, None

    major, minor = torch.cuda.get_device_capability(0)
    capability = f"{major}.{minor}"
    device_name = torch.cuda.get_device_name(0)
    sm_tag = f"sm_{major}{minor}"

    # torch.cuda._get_arch_list() returns the SM architectures this build
    # was compiled for (e.g. ["sm_50", "sm_60", ..., "sm_90"]).
    try:
        arch_list = torch.cuda._get_arch_list()
        if arch_list:
            # Check for both sm_XX and compute_XX (JIT-compiled) entries
            compute_tag = f"compute_{major}{minor}"
            if sm_tag not in arch_list and compute_tag not in arch_list:
                return False, (
                    f"{device_name} (compute capability {capability} / {sm_tag}) "
                    f"is not supported by this PyTorch build. "
                    f"Supported architectures: {', '.join(arch_list)}. "
                    f"Install PyTorch nightly (cu128) for newer GPU support: "
                    f"pip install torch --index-url https://download.pytorch.org/whl/nightly/cu128"
                )
    except AttributeError:
        pass

    return True, None


def empty_device_cache(device: str) -> None:
    """
    Free cached memory on the given device (CUDA or XPU).

    Backends should call this after unloading models so VRAM is returned
    to the OS.
    """
    import torch

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()


def manual_seed(seed: int, device: str) -> None:
    """
    Set the random seed on both CPU and the active accelerator.

    Covers CUDA and Intel XPU so that generation is reproducible
    regardless of which GPU backend is in use.
    """
    import torch

    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    elif device == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.manual_seed(seed)


async def combine_voice_prompts(
    audio_paths: List[str],
    reference_texts: List[str],
    *,
    sample_rate: Optional[int] = None,
) -> Tuple[np.ndarray, str]:
    """
    Combine multiple reference audio samples into one.

    Loads each audio file, normalizes, concatenates, and joins texts.

    Args:
        audio_paths: Paths to reference audio files.
        reference_texts: Corresponding transcripts.
        sample_rate: If set, resample audio to this rate during loading.
    """
    combined_audio = []

    for path in audio_paths:
        kwargs = {"sample_rate": sample_rate} if sample_rate else {}
        audio, _sr = load_audio(path, **kwargs)
        audio = normalize_audio(audio)
        combined_audio.append(audio)

    mixed = np.concatenate(combined_audio)
    mixed = normalize_audio(mixed)
    combined_text = " ".join(reference_texts)

    return mixed, combined_text


@contextmanager
def model_load_progress(
    model_name: str,
    is_cached: bool,
    filter_non_downloads: Optional[bool] = None,
):
    """
    Context manager for model loading with HF download progress tracking.

    Handles the tqdm patching, progress_manager/task_manager lifecycle,
    and error reporting that every backend duplicates.

    Args:
        model_name: Progress tracking key (e.g. "qwen-tts-1.7B", "whisper-base").
        is_cached: Whether the model is already downloaded.
        filter_non_downloads: Whether to filter non-download tqdm bars.
                              Defaults to `is_cached`.

    Yields:
        The tracker context (already entered). The caller loads the model
        inside the `with` block. The tqdm patch is torn down on exit.

    Usage:
        with model_load_progress("qwen-tts-1.7B", is_cached) as ctx:
            self.model = SomeModel.from_pretrained(...)
    """
    if filter_non_downloads is None:
        filter_non_downloads = is_cached

    progress_manager = get_progress_manager()
    task_manager = get_task_manager()

    progress_callback = create_hf_progress_callback(model_name, progress_manager)
    tracker = HFProgressTracker(progress_callback, filter_non_downloads=filter_non_downloads)

    tracker_context = tracker.patch_download()
    tracker_context.__enter__()

    if not is_cached:
        task_manager.start_download(model_name)
        progress_manager.update_progress(
            model_name=model_name,
            current=0,
            total=0,
            filename="Connecting to HuggingFace...",
            status="downloading",
        )

    try:
        yield tracker_context
    except Exception as e:
        # Report error to both managers
        progress_manager.mark_error(model_name, str(e))
        task_manager.error_download(model_name, str(e))
        raise
    else:
        # Only mark complete if we were tracking a download
        if not is_cached:
            progress_manager.mark_complete(model_name)
            task_manager.complete_download(model_name)
    finally:
        tracker_context.__exit__(None, None, None)


def patch_chatterbox_f32(model) -> None:
    """
    Patch float64 -> float32 dtype mismatches in upstream chatterbox.

    librosa.load returns float64 numpy arrays. Multiple upstream code paths
    convert these to torch tensors via torch.from_numpy() without casting,
    then matmul against float32 model weights. This patches the two known
    entry points:

    1. S3Tokenizer.log_mel_spectrogram — audio tensor hits _mel_filters (f32)
    2. VoiceEncoder.forward — float64 mel spectrograms hit LSTM weights (f32)
    """
    import types

    # Patch S3Tokenizer
    _tokzr = model.s3gen.tokenizer
    _orig_log_mel = _tokzr.log_mel_spectrogram.__func__

    def _f32_log_mel(self_tokzr, audio, padding=0):
        import torch as _torch

        if _torch.is_tensor(audio):
            audio = audio.float()
        return _orig_log_mel(self_tokzr, audio, padding)

    _tokzr.log_mel_spectrogram = types.MethodType(_f32_log_mel, _tokzr)

    # Patch VoiceEncoder
    _ve = model.ve
    _orig_ve_forward = _ve.forward.__func__

    def _f32_ve_forward(self_ve, mels):
        return _orig_ve_forward(self_ve, mels.float())

    _ve.forward = types.MethodType(_f32_ve_forward, _ve)
