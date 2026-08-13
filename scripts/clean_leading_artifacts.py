#!/usr/bin/env python3
"""Limpia el artefacto "beat/noa" del inicio de audios TTS (voicebox).

Uso:
    backend/venv/bin/python scripts/clean_leading_artifacts.py <wav1> [wav2 ...]

Cada archivo se limpia IN-PLACE y el original se respalda como <name>_orig.wav.
"""
import sys, os, wave
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from utils.audio import trim_leading_artifact

def load(path):
    w = wave.open(path, 'rb')
    sr = w.getframerate()
    data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)/32768
    return data, sr

def save(audio, sr, path):
    w = wave.open(path, 'wb')
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    w.close()

def main():
    if len(sys.argv) < 2:
        print(__doc__); return 1
    for path in sys.argv[1:]:
        if not os.path.exists(path):
            print(f"⚠ no existe: {path}"); continue
        a, sr = load(path)
        before = len(a) / sr
        t = trim_leading_artifact(a, sr)
        if len(t) == len(a):
            print(f"• {os.path.basename(path)}: sin artefacto, sin cambios")
            continue
        backup = path.replace('.wav', '_orig.wav')
        os.rename(path, backup)
        save(t, sr, path)
        print(f"✓ {os.path.basename(path)}: {before:.2f}s -> {len(t)/sr:.2f}s (cortó {before-len(t)/sr:.3f}s, backup {os.path.basename(backup)})")
    return 0

if __name__ == '__main__':
    sys.exit(main())
