"""
Prepare audio data for Whisper LoRA fine-tuning.

Input: a directory of WAV files + a manifest (JSON or JSONL).
Output: a HuggingFace DatasetDict saved to disk.

Usage:
    python prepare_data.py \
        --audio_dir ./data/audio \
        --manifest ./data/manifest.json \
        --output_dir ./data/prepared \
        --language he
"""

import argparse
import json
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict


def load_manifest(manifest_path, audio_dir):
    """Load manifest from JSON dict or JSONL file."""
    manifest_path = Path(manifest_path)
    audio_dir = Path(audio_dir)
    entries = []

    with open(manifest_path) as f:
        content = f.read().strip()

    # Try JSON dict format: {"filename.wav": "transcription", ...}
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            for filename, text in data.items():
                audio_path = audio_dir / filename
                if audio_path.exists():
                    entries.append({"audio": str(audio_path), "text": text})
                else:
                    print(f"  Warning: {audio_path} not found, skipping")
            return entries
    except json.JSONDecodeError:
        pass

    # Try JSONL format: {"audio": "filename.wav", "text": "transcription"}
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        audio_path = audio_dir / item["audio"]
        if audio_path.exists():
            entries.append({"audio": str(audio_path), "text": item["text"]})
        else:
            print(f"  Warning: {audio_path} not found, skipping")

    return entries


def process_audio(audio_path, target_sr=16000, max_duration=30.0):
    """Load and validate a single audio file."""
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        duration = len(audio) / sr

        if duration < 0.5:
            print(f"  Skipping {audio_path}: too short ({duration:.1f}s)")
            return None

        if duration > max_duration:
            audio = audio[: int(max_duration * sr)]
            duration = max_duration

        return {"audio": audio, "sr": sr, "duration": duration}
    except Exception as e:
        print(f"  Error loading {audio_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Prepare data for Whisper LoRA fine-tuning")
    parser.add_argument("--audio_dir", required=True, help="Directory containing WAV files")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json or manifest.jsonl")
    parser.add_argument("--output_dir", required=True, help="Output directory for prepared dataset")
    parser.add_argument("--test_split", type=float, default=0.1, help="Fraction for test set")
    parser.add_argument("--language", default="he", help="Language code")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading manifest from {args.manifest}...")
    entries = load_manifest(args.manifest, args.audio_dir)
    print(f"  Found {len(entries)} entries")

    if not entries:
        print("Error: No valid entries found!")
        return

    valid_entries = []
    total_duration = 0.0

    for entry in entries:
        result = process_audio(entry["audio"])
        if result:
            valid_entries.append({
                "audio": entry["audio"],
                "text": entry["text"],
                "duration": result["duration"],
                "language": args.language,
            })
            total_duration += result["duration"]

    print(f"  Valid entries: {len(valid_entries)} ({total_duration / 60:.1f} minutes)")

    random.shuffle(valid_entries)
    split_idx = max(1, int(len(valid_entries) * (1 - args.test_split)))
    train_entries = valid_entries[:split_idx]
    test_entries = valid_entries[split_idx:]

    print(f"  Train: {len(train_entries)}, Test: {len(test_entries)}")

    def make_dataset(entries):
        return Dataset.from_dict({
            "audio": [e["audio"] for e in entries],
            "text": [e["text"] for e in entries],
            "language": [e["language"] for e in entries],
        }).cast_column("audio", Audio(sampling_rate=16000))

    ds = DatasetDict({
        "train": make_dataset(train_entries),
        "test": make_dataset(test_entries),
    })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))

    print(f"\nDone! Total audio: {total_duration / 60:.1f} minutes")
    print(f"  Train: {len(train_entries)} samples")
    print(f"  Test:  {len(test_entries)} samples")
    print(f"  Saved to {output_dir}")


if __name__ == "__main__":
    main()
