#!/usr/bin/env python3
"""
Inference with a LoRA-adapted Whisper model.

Supports two modes:
  1. Load adapter on-the-fly (no merge needed, requires peft)
  2. Load a pre-merged model (no peft dependency)

Usage:
    # With adapter (no merge)
    python inference_lora.py \
        --mode adapter \
        --base_model ivrit-ai/whisper-large-v3-turbo \
        --adapter_path ./checkpoints/whisper-lora/final-adapter \
        --audio_path ./test.wav \
        --language he

    # With merged model
    python inference_lora.py \
        --mode merged \
        --model_path ./models/whisper-lora-merged \
        --audio_path ./test.wav \
        --language he

    # Batch transcription from a directory
    python inference_lora.py \
        --mode merged \
        --model_path ./models/whisper-lora-merged \
        --audio_dir ./audio_files/ \
        --language he \
        --output results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import librosa
import numpy as np


def get_device() -> str:
    """Select best device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS works better for inference than training
        return "mps"
    return "cpu"


def load_model_adapter(base_model: str, adapter_path: str, device: str):
    """Load base model + LoRA adapter (requires peft)."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel

    print(f"Loading base model: {base_model}")
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.to(device)
    model.eval()

    # Load processor from adapter path or base model
    try:
        processor = WhisperProcessor.from_pretrained(adapter_path)
    except Exception:
        processor = WhisperProcessor.from_pretrained(base_model)

    return model, processor


def load_model_merged(model_path: str, device: str):
    """Load pre-merged model (no peft dependency)."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    print(f"Loading merged model: {model_path}")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()

    processor = WhisperProcessor.from_pretrained(model_path)

    return model, processor


def transcribe_audio(
    model, processor, audio_path: str, language: str, device: str
) -> dict:
    """Transcribe a single audio file."""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio) / 16000

    # Process
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.to(device)

    # Generate
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs["input_features"],
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=225,
        )

    # Decode
    transcription = processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0].strip()

    return {
        "file": os.path.basename(audio_path),
        "path": str(audio_path),
        "duration_seconds": round(duration, 2),
        "transcription": transcription,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inference with LoRA-adapted Whisper model"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["adapter", "merged"],
        default="merged",
        help="Loading mode: 'adapter' (base + LoRA) or 'merged' (pre-merged model)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="ivrit-ai/whisper-large-v3-turbo",
        help="Base model ID (only for adapter mode)",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter (only for adapter mode)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to merged model (only for merged mode)",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Path to a single audio file to transcribe",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default=None,
        help="Directory of audio files to batch-transcribe",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="he",
        help="Language code (default: he)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for batch results",
    )

    args = parser.parse_args()

    # Validate args
    if args.mode == "adapter" and not args.adapter_path:
        parser.error("--adapter_path is required in adapter mode")
    if args.mode == "merged" and not args.model_path:
        parser.error("--model_path is required in merged mode")
    if not args.audio_path and not args.audio_dir:
        parser.error("Provide --audio_path or --audio_dir")

    # Select device
    device = get_device()
    print(f"Device: {device}")

    # Load model
    if args.mode == "adapter":
        model, processor = load_model_adapter(
            args.base_model, args.adapter_path, device
        )
    else:
        model, processor = load_model_merged(args.model_path, device)

    # Collect audio files
    audio_files = []
    if args.audio_path:
        audio_files.append(args.audio_path)
    if args.audio_dir:
        audio_dir = Path(args.audio_dir)
        for ext in ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"]:
            audio_files.extend(sorted(audio_dir.glob(ext)))

    if not audio_files:
        print("No audio files found!")
        sys.exit(1)

    print(f"\nTranscribing {len(audio_files)} file(s)...\n")

    # Transcribe
    results = []
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] {audio_file}")
        result = transcribe_audio(model, processor, str(audio_file), args.language, device)
        results.append(result)
        print(f"  -> {result['transcription'][:100]}{'...' if len(result['transcription']) > 100 else ''}")

    # Save results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Print summary
    if len(results) > 1:
        total_duration = sum(r["duration_seconds"] for r in results)
        print(f"\nSummary:")
        print(f"  Files processed: {len(results)}")
        print(f"  Total audio: {total_duration:.1f}s ({total_duration / 60:.1f} min)")


if __name__ == "__main__":
    main()
