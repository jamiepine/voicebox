"""
Chatterbox Multilingual TTS backend for Hebrew voice generation.

Uses ChatterboxMultilingualTTS from the chatterbox-tts package
(ResembleAI/chatterbox on HuggingFace) which supports 23 languages
including Hebrew. Forces CPU on macOS due to known MPS tensor issues.
"""

from typing import Optional, List, Tuple
import asyncio
import platform
import numpy as np
from pathlib import Path

from . import TTSBackend
from ..utils.audio import normalize_audio, load_audio
from ..utils.progress import get_progress_manager
from ..utils.hf_progress import HFProgressTracker, create_hf_progress_callback
from ..utils.tasks import get_task_manager


class ChatterboxTTSBackend:
    """Chatterbox Multilingual TTS backend for Hebrew voice cloning."""

    HF_REPO_ID = "ResembleAI/chatterbox"

    # Files specific to the multilingual model
    _MTL_WEIGHT_FILES = [
        "t3_mtl23ls_v2.safetensors",
        "s3gen.pt",
        "ve.pt",
    ]

    def __init__(self):
        self.model = None
        self._device = None

    def _get_device(self) -> str:
        """Get the best available device. Forces CPU on macOS (MPS issue)."""
        if platform.system() == "Darwin":
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        """Get model path (Chatterbox has a single model)."""
        return self.HF_REPO_ID

    def _is_model_cached(self, model_size: str) -> bool:
        """Check if the Chatterbox multilingual model is cached locally."""
        try:
            from huggingface_hub import constants as hf_constants
            repo_cache = Path(hf_constants.HF_HUB_CACHE) / (
                "models--" + self.HF_REPO_ID.replace("/", "--")
            )

            if not repo_cache.exists():
                return False

            blobs_dir = repo_cache / "blobs"
            if blobs_dir.exists() and any(blobs_dir.glob("*.incomplete")):
                return False

            # Check specifically for multilingual weight files
            snapshots_dir = repo_cache / "snapshots"
            if snapshots_dir.exists():
                for fname in self._MTL_WEIGHT_FILES:
                    if not any(snapshots_dir.rglob(fname)):
                        return False
                return True

            return False
        except Exception as e:
            print(f"[ChatterboxTTSBackend._is_model_cached] Error: {e}")
            return False

    async def load_model(self, model_size: str = "default") -> None:
        """Load the Chatterbox multilingual model."""
        if self.model is not None:
            return
        await asyncio.to_thread(self._load_model_sync, model_size)

    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        try:
            progress_manager = get_progress_manager()
            task_manager = get_task_manager()
            progress_model_name = "chatterbox-tts"

            is_cached = self._is_model_cached(model_size)

            progress_callback = create_hf_progress_callback(
                progress_model_name, progress_manager
            )
            tracker = HFProgressTracker(
                progress_callback, filter_non_downloads=is_cached
            )

            if not is_cached:
                task_manager.start_download(progress_model_name)
                progress_manager.update_progress(
                    model_name=progress_model_name,
                    current=0,
                    total=0,
                    filename="Connecting to HuggingFace...",
                    status="downloading",
                )

            with tracker.patch_download():
                device = self._get_device()
                self._device = device

                print(f"Loading Chatterbox Multilingual TTS model on {device}...")

                import torch
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS

                # Patch Hebrew diacritization: dicta-onnx needs a model_path
                # argument in newer versions. Patch the tokenizer module so
                # Chatterbox can find and use the dicta ONNX model.
                self._patch_hebrew_diacritics()

                # Monkey-patch torch.load to add map_location for CPU loading.
                # The multilingual model's .pt files were saved on CUDA and
                # from_local() doesn't pass map_location, so loading on CPU fails.
                if device == "cpu":
                    import threading
                    _orig_torch_load = torch.load
                    _load_lock = threading.Lock()

                    def _patched_load(*args, **kwargs):
                        kwargs.setdefault("map_location", "cpu")
                        return _orig_torch_load(*args, **kwargs)

                    with _load_lock:
                        torch.load = _patched_load
                        try:
                            self.model = ChatterboxMultilingualTTS.from_pretrained(device=device)
                        finally:
                            torch.load = _orig_torch_load
                else:
                    self.model = ChatterboxMultilingualTTS.from_pretrained(device=device)

                # Fix: newer transformers defaults LlamaModel to sdpa attention
                # which doesn't support output_attentions (needed by Chatterbox
                # AlignmentStreamAnalyzer). Force eager attention instead.
                t3_tfmr = self.model.t3.tfmr
                if hasattr(t3_tfmr, 'config') and hasattr(t3_tfmr.config, '_attn_implementation'):
                    t3_tfmr.config._attn_implementation = "eager"
                    # Also update each layer's self_attn
                    for layer in getattr(t3_tfmr, 'layers', []):
                        if hasattr(layer, 'self_attn'):
                            layer.self_attn._attn_implementation = "eager"

            if not is_cached:
                progress_manager.mark_complete(progress_model_name)
                task_manager.complete_download(progress_model_name)

            print("Chatterbox Multilingual TTS model loaded successfully")

        except ImportError as e:
            print(
                "Error: chatterbox-tts package not found. "
                "Install with: pip install chatterbox-tts"
            )
            progress_manager = get_progress_manager()
            task_manager = get_task_manager()
            progress_manager.mark_error("chatterbox-tts", str(e))
            task_manager.error_download("chatterbox-tts", str(e))
            raise
        except Exception as e:
            print(f"Error loading Chatterbox Multilingual TTS model: {e}")
            progress_manager = get_progress_manager()
            task_manager = get_task_manager()
            progress_manager.mark_error("chatterbox-tts", str(e))
            task_manager.error_download("chatterbox-tts", str(e))
            raise

    @staticmethod
    def _patch_hebrew_diacritics():
        """Patch chatterbox tokenizer to use local dicta-onnx model for Hebrew nikud."""
        try:
            dicta_model_path = str(
                Path(__file__).resolve().parent.parent / "models" / "dicta-onnx" / "dicta-1.0.onnx"
            )
            if not Path(dicta_model_path).exists():
                print(f"[Chatterbox] dicta model not found at {dicta_model_path}, Hebrew diacritics disabled")
                return

            from chatterbox.models.tokenizers import tokenizer as tok_mod
            from dicta_onnx import Dicta

            _dicta_instance = Dicta(dicta_model_path)

            def _patched_add_hebrew_diacritics(text: str) -> str:
                try:
                    return _dicta_instance.add_diacritics(text)
                except Exception as e:
                    print(f"[Chatterbox] Hebrew diacritization failed: {e}")
                    return text

            tok_mod.add_hebrew_diacritics = _patched_add_hebrew_diacritics
            print("[Chatterbox] Hebrew diacritization (nikud) enabled via dicta-onnx")
        except Exception as e:
            print(f"[Chatterbox] Could not patch Hebrew diacritics: {e}")

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            device = self._device
            del self.model
            self.model = None
            self._device = None
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()
            print("Chatterbox Multilingual TTS model unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        Chatterbox processes reference audio at generation time,
        so the prompt simply stores the file path and text.
        """
        voice_prompt = {
            "ref_audio": str(audio_path),
            "ref_text": reference_text,
        }
        return voice_prompt, False

    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        """Combine multiple reference samples."""
        combined_audio = []

        for audio_path in audio_paths:
            audio, _sr = load_audio(audio_path)
            audio = normalize_audio(audio)
            combined_audio.append(audio)

        mixed = np.concatenate(combined_audio)
        mixed = normalize_audio(mixed)
        combined_text = " ".join(reference_texts)

        return mixed, combined_text

    # Tuned generation defaults per language.
    # Lower temperature + higher cfg_weight = clearer pronunciation.
    _LANG_DEFAULTS = {
        "he": {"exaggeration": 0.4, "cfg_weight": 0.7, "temperature": 0.65, "repetition_penalty": 2.5},
    }
    _GLOBAL_DEFAULTS = {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8, "repetition_penalty": 2.0}

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "he",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
        *,
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate multilingual audio using Chatterbox.

        Args:
            text: Text to synthesize
            voice_prompt: Dict with ref_audio path
            language: Language code (e.g. "he" for Hebrew)
            seed: Random seed for reproducibility
            instruct: Unused (kept for protocol compatibility)
            exaggeration: Emotion intensity (0-1, default varies by lang)
            cfg_weight: Classifier-free guidance (higher = more faithful)
            temperature: Sampling temperature (lower = more deterministic)
            repetition_penalty: Token repetition penalty

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        await self.load_model()

        ref_audio = voice_prompt.get("ref_audio")
        if ref_audio and not Path(ref_audio).exists():
            print(f"Warning: Reference audio not found: {ref_audio}")
            ref_audio = None

        # Merge language-specific defaults with any explicit overrides
        lang_defaults = self._LANG_DEFAULTS.get(language, self._GLOBAL_DEFAULTS)
        gen_params = {
            "exaggeration": exaggeration if exaggeration is not None else lang_defaults.get("exaggeration", 0.5),
            "cfg_weight": cfg_weight if cfg_weight is not None else lang_defaults.get("cfg_weight", 0.5),
            "temperature": temperature if temperature is not None else lang_defaults.get("temperature", 0.8),
            "repetition_penalty": repetition_penalty if repetition_penalty is not None else lang_defaults.get("repetition_penalty", 2.0),
        }

        def _generate_sync():
            import torch

            if seed is not None:
                torch.manual_seed(seed)

            print(f"[Chatterbox] Generating: lang={language}, params={gen_params}")

            wav = self.model.generate(
                text,
                language_id=language,
                audio_prompt_path=ref_audio,
                **gen_params,
            )

            # Chatterbox returns a torch tensor; convert to numpy
            if isinstance(wav, torch.Tensor):
                audio = wav.squeeze().cpu().numpy().astype(np.float32)
            else:
                audio = np.asarray(wav, dtype=np.float32)

            sample_rate = getattr(self.model, 'sr', None) or getattr(self.model, 'sample_rate', 24000)

            return audio, sample_rate

        return await asyncio.to_thread(_generate_sync)
