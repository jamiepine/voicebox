"""
MiniCPM5-1B LLM backend implementations.

Mirrors qwen_llm_backend.py's shape (MLX and PyTorch paths sharing the same
`LLMBackend` protocol and model-load progress plumbing) — MiniCPM5-1B is a
plain LlamaForCausalLM checkpoint whose chat template accepts the same
enable_thinking kwarg Qwen3's does, so the load/generate logic below is a
direct mirror rather than a new design.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from . import LLMBackend, DEFAULT_LLM_MAX_TOKENS, DEFAULT_LLM_TEMPERATURE
from .base import (
    is_model_cached,
    get_torch_device,
    empty_device_cache,
    empty_mlx_cache,
    model_load_progress,
)
from .qwen_llm_backend import _build_messages

logger = logging.getLogger(__name__)


PYTORCH_HF_REPOS = {
    "1B": "openbmb/MiniCPM5-1B",
}

# Official pre-quantized 4-bit weights only — no community quant, per
# explicit product decision (see specs/001-minicpm5-llm-engine/research.md).
MLX_HF_REPOS = {
    "1B": "openbmb/MiniCPM5-1B-MLX",
}


def _ensure_compatible_mlx_snapshot(local_snapshot_dir: str) -> str:
    """Work around openbmb/MiniCPM5-1B-MLX declaring a tokenizer_class this
    project's pinned transformers version doesn't have.

    The repo's tokenizer_config.json says "tokenizer_class": "TokenizersBackend"
    (a transformers-5.x-era name); this project pins transformers<=4.57.6
    project-wide (mlx-audio's TTS/STT path needs that older API surface — see
    backend/requirements-mlx.txt), so AutoTokenizer.from_pretrained refuses to
    load it at all. The underlying tokenizer.json is a standard fast-tokenizer
    file — only the declared class name is stale — so this builds a staging
    directory that symlinks every file except a patched tokenizer_config.json,
    letting mlx_lm.load() succeed without duplicating the multi-hundred-MB
    weight file on disk. Confirmed via direct testing against the live repo.
    """
    import tempfile

    import transformers

    src = Path(local_snapshot_dir)
    config_path = src / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    declared_class = config.get("tokenizer_class")

    if declared_class is None or hasattr(transformers, declared_class):
        return local_snapshot_dir

    staging_dir = Path(tempfile.mkdtemp(prefix="minicpm5-mlx-tokenizer-fix-"))
    for entry in src.iterdir():
        if entry.name == "tokenizer_config.json":
            continue
        (staging_dir / entry.name).symlink_to(entry.resolve())

    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    (staging_dir / "tokenizer_config.json").write_text(json.dumps(config))
    return str(staging_dir)


class PyTorchMiniCPMLLMBackend:
    """MiniCPM5-1B LLM backend using HuggingFace transformers."""

    def __init__(self, model_size: str = "1B"):
        self.model = None
        self.tokenizer = None
        self.model_size = model_size
        self._current_model_size: Optional[str] = None
        self.device = self._get_device()

    def _get_device(self) -> str:
        return get_torch_device(allow_xpu=True, allow_directml=True, allow_mps=True)

    def is_loaded(self) -> bool:
        """Whether the model is currently loaded in memory."""
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        if model_size not in PYTORCH_HF_REPOS:
            raise ValueError(f"Unknown MiniCPM5 size: {model_size}")
        return PYTORCH_HF_REPOS[model_size]

    def _is_model_cached(self, model_size: str) -> bool:
        return is_model_cached(self._get_model_path(model_size))

    async def load_model(self, model_size: Optional[str] = None) -> None:
        """Load the given size (default: the instance's own), swapping out any other loaded size first."""
        if model_size is None:
            model_size = self.model_size

        if self.model is not None and self._current_model_size == model_size:
            return

        if self.model is not None and self._current_model_size != model_size:
            self.unload_model()

        await asyncio.to_thread(self._load_model_sync, model_size)

    def _load_model_sync(self, model_size: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        progress_model_name = f"minicpm5-{model_size.lower()}"
        is_cached = self._is_model_cached(model_size)
        repo = self._get_model_path(model_size)

        with model_load_progress(progress_model_name, is_cached):
            logger.info("Loading MiniCPM5 %s on %s...", model_size, self.device)
            # See qwen_llm_backend.py's equivalent comment (issue #841) — no
            # offline forcing here either.
            self.tokenizer = AutoTokenizer.from_pretrained(repo)
            dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                repo,
                dtype=dtype,
            )
            self.model.to(self.device)
            self.model.eval()

        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("MiniCPM5 %s loaded successfully", model_size)

    def unload_model(self) -> None:
        """Drop the loaded model/tokenizer and release the memory they held."""
        if self.model is None:
            return
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._current_model_size = None
        empty_device_cache(self.device)
        logger.info("MiniCPM5 unloaded")

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        model_size: Optional[str] = None,
        examples: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        """Load the model if needed and run a single-turn chat completion."""
        await self.load_model(model_size)
        return await asyncio.to_thread(
            self._generate_sync, prompt, system, max_tokens, temperature, examples
        )

    def _generate_sync(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        examples: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        import torch

        messages = _build_messages(prompt, system, examples)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        do_sample = temperature > 0
        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = 0.9

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0, input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class MLXMiniCPMLLMBackend:
    """MiniCPM5-1B LLM backend using mlx-lm (Apple Silicon)."""

    def __init__(self, model_size: str = "1B"):
        self.model = None
        self.tokenizer = None
        self.model_size = model_size
        self._current_model_size: Optional[str] = None

    def is_loaded(self) -> bool:
        """Whether the model is currently loaded in memory."""
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        if model_size not in MLX_HF_REPOS:
            raise ValueError(f"Unknown MiniCPM5 size: {model_size}")
        return MLX_HF_REPOS[model_size]

    def _is_model_cached(self, model_size: str) -> bool:
        return is_model_cached(
            self._get_model_path(model_size),
            weight_extensions=(".safetensors", ".bin", ".npz"),
        )

    async def load_model(self, model_size: Optional[str] = None) -> None:
        """Load the given size (default: the instance's own), swapping out any other loaded size first."""
        if model_size is None:
            model_size = self.model_size

        if self.model is not None and self._current_model_size == model_size:
            return

        if self.model is not None and self._current_model_size != model_size:
            self.unload_model()

        await asyncio.to_thread(self._load_model_sync, model_size)

    def _load_model_sync(self, model_size: str) -> None:
        from huggingface_hub import snapshot_download
        from mlx_lm import load as mlx_load

        progress_model_name = f"minicpm5-{model_size.lower()}"
        is_cached = self._is_model_cached(model_size)
        repo = self._get_model_path(model_size)

        with model_load_progress(progress_model_name, is_cached):
            logger.info("Loading MiniCPM5 %s via MLX...", model_size)
            # snapshot_download first (not mlx_load(repo) directly) so we can
            # patch the local tokenizer_config.json before mlx_lm reads it —
            # see _ensure_compatible_mlx_snapshot. Idempotent/cached, so this
            # doesn't cause a second download.
            local_dir = snapshot_download(repo)
            compatible_dir = _ensure_compatible_mlx_snapshot(local_dir)
            loaded = mlx_load(compatible_dir)

        # mlx_lm.load returns (model, tokenizer) by default and
        # (model, tokenizer, config) when return_config=True.
        self.model = loaded[0]
        self.tokenizer = loaded[1]

        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("MiniCPM5 %s (MLX) loaded successfully", model_size)

    def unload_model(self) -> None:
        """Drop the loaded model/tokenizer and release the memory they held."""
        if self.model is None:
            return
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._current_model_size = None
        empty_mlx_cache()
        logger.info("MiniCPM5 (MLX) unloaded")

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        model_size: Optional[str] = None,
        examples: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        """Load the model if needed and run a single-turn chat completion."""
        await self.load_model(model_size)
        return await asyncio.to_thread(
            self._generate_sync, prompt, system, max_tokens, temperature, examples
        )

    def _generate_sync(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        examples: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        messages = _build_messages(prompt, system, examples)
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        sampler = make_sampler(temp=temperature, top_p=0.9) if temperature > 0 else None
        text = mlx_generate(
            self.model,
            self.tokenizer,
            prompt=chat_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return text.strip()
