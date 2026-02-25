"""
Merge LoRA adapter into base Whisper model and export.

Usage:
    python merge_adapter.py \
        --base_model ivrit-ai/whisper-large-v3-turbo \
        --adapter_path ./checkpoints/whisper-lora/final-adapter \
        --output_dir ./models/whisper-lora-merged

    # With test + float16:
    python merge_adapter.py ... --test_audio ./test.wav --language he --save_float16
"""

import argparse
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into Whisper")
    parser.add_argument("--base_model", default="ivrit-ai/whisper-large-v3-turbo")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_float16", action="store_true")
    parser.add_argument("--test_audio", help="WAV file to test after merging")
    parser.add_argument("--language", default="he")
    parser.add_argument("--push_to_hub", help="HuggingFace Hub repo (e.g. user/model)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {args.base_model}...")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.float32
    )
    processor = WhisperProcessor.from_pretrained(args.base_model)

    print(f"Loading adapter: {args.adapter_path}...")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print("Merging...")
    model = model.merge_and_unload()

    print(f"Saving to {output_dir}...")
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    size_mb = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"  Size: {size_mb:.0f} MB")

    if args.save_float16:
        fp16_dir = output_dir / "float16"
        fp16_dir.mkdir(exist_ok=True)
        print(f"Saving float16 to {fp16_dir}...")
        model.half().save_pretrained(str(fp16_dir))
        processor.save_pretrained(str(fp16_dir))
        model.float()

    if args.test_audio:
        print(f"\nTesting: {args.test_audio}")
        import librosa
        audio, sr = librosa.load(args.test_audio, sr=16000, mono=True)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        forced = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")
        with torch.no_grad():
            ids = model.generate(inputs["input_features"], forced_decoder_ids=forced)
        print(f"  Result: {processor.batch_decode(ids, skip_special_tokens=True)[0]}")

    if args.push_to_hub:
        print(f"\nPushing to {args.push_to_hub}...")
        model.push_to_hub(args.push_to_hub)
        processor.push_to_hub(args.push_to_hub)

    print("\nDone!")


if __name__ == "__main__":
    main()
