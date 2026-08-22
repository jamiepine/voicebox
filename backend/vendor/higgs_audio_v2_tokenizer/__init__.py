"""Higgs Audio V2 neural audio codec, vendored from transformers 5.x.

OmniVoice imports ``HiggsAudioV2TokenizerModel`` straight from the
``transformers`` namespace, but the class only landed in transformers 5.3 and
Voicebox is capped at 4.57.6. The weights ship inside the OmniVoice repo
(``audio_tokenizer/model.safetensors``); only the modelling code is missing,
so we vendor it.

Regenerate with ``python scripts/sync-higgs-tokenizer.py``. Install into the
``transformers`` namespace with
``backend.utils.transformers5_compat.install()`` -- that must run
before ``omnivoice`` is imported.
"""

from .configuration_higgs_audio_v2_tokenizer import HiggsAudioV2TokenizerConfig
from .modeling_higgs_audio_v2_tokenizer import (
    HiggsAudioV2TokenizerModel,
    HiggsAudioV2TokenizerPreTrainedModel,
)

__all__ = [
    "HiggsAudioV2TokenizerConfig",
    "HiggsAudioV2TokenizerModel",
    "HiggsAudioV2TokenizerPreTrainedModel",
]
