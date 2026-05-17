# Denoiser Integration Notes

## Objective

Add an optional voice denoising layer to Voicebox without breaking the current
TTS, CUDA, SRT2Voice, or voice cloning workflows.

The denoiser is useful in two places:

- SRT2Voice full narration cleanup before Whisper alignment, RMS/ZCR analysis,
  Auto Cut, and timeline export.
- Voice creation from user audio samples, where cleaner reference audio can
  improve cloning quality and reduce breath/noise artifacts.

The denoiser must never become a hidden mandatory dependency. If the denoise
backend is missing, Voicebox must continue to work exactly as before.

## Current Decision

Preferred denoise candidate: DeepFilterNet3.

Reason:

- DNSMOS evaluates audio quality but does not denoise.
- DeepFilterNet3 is designed for speech enhancement and residual noise
  suppression.
- User testing in Audacity with the OpenVINO denoiser using DeepFilterNet3 gave
  excellent results.
- DeepFilterNet3 appears better suited than generic gates or spectral filters
  because it can reduce breath/noise while preserving speech tails.

## What Was Tested

The embedded Voicebox Python environment currently has:

- `onnxruntime` available
- `soundfile` available
- `librosa` available
- no `openvino`
- no usable `df` / DeepFilterNet runtime
- no `libdf`

`DeepFilterNet-py312` was tested because Voicebox runs on Python 3.12. The
package installed the Python `df` module, but failed at runtime because the
native `libdf` module was missing. No matching `libdf` package was available via
pip for this Windows/Python 3.12 environment.

Conclusion: do not force this package into the main venv.

## DNSMOS Clarification

The Microsoft DNS-Challenge `DNSMOS/DNSMOS` folder contains ONNX models such as:

- `sig.onnx`
- `bak_ovr.onnx`
- `sig_bak_ovr.onnx`
- `model_v8.onnx`

These models estimate audio quality. They are not denoisers.

Possible future use:

- Compare raw vs denoised quality.
- Generate a `quality_report.json` in SRT2Voice export packages.
- Reject or warn about denoise output if speech quality degrades.

DNSMOS should be treated as QA, not cleanup.

## SRT2Voice Integration Target

The intended SRT2Voice chain is:

```text
Qwen full narration raw WAV
-> optional denoise
-> full narration denoised WAV
-> Whisper word alignment
-> RMS/ZCR acoustic boundary analysis
-> Auto Cut
-> timeline export
```

Rules:

- Denoise must run before Whisper and RMS/ZCR.
- The raw full narration WAV must remain available for rollback and comparison.
- The denoised WAV becomes the analysis and timeline source only when denoise
  succeeds.
- If denoise fails or backend is unavailable, the raw WAV remains the source.
- The export package should include raw, denoised if available, and debug
  metadata.

Suggested generated files:

```text
generations/dubbing_cuts/<project_id>/full_narration_raw.wav
generations/dubbing_cuts/<project_id>/full_narration_denoised.wav
generations/dubbing_cuts/<project_id>/denoise_debug.json
```

Suggested export package paths:

```text
audio/full_narration_raw.wav
audio/full_narration_denoised.wav
debug/denoise_debug.json
```

## Voice Creation Integration Target

Denoise can also help when creating voices from audio samples.

Potential workflow:

```text
User sample WAV
-> optional preview denoise
-> cloned voice reference audio
```

Rules:

- The user must be able to keep the original sample.
- Denoised samples should be explicit, not silent destructive replacements.
- The denoise step should preserve duration and speech timing.
- Voice cloning should be able to use either original or denoised sample.
- For cloned voices, denoise may improve timbre cleanliness but will not give
  reliable delivery instruction control. Prosody still mainly follows reference
  audio.

## Why Not a Simple Gate First

FFmpeg `agate` or similar gates can create clean silence, but they may also cut:

- weak French final consonants
- nasal tails
- breathy endings
- low-energy speech tails

This directly conflicts with the Auto Cut goal of preserving the true acoustic
end of words. A gate may be useful later as an optional second stage, but not as
the first denoise implementation.

## Packaging Direction

DeepFilterNet3 should be packaged as an optional backend, not as an implicit
runtime dependency.

Preferred packaging model:

- Similar spirit to the CUDA backend: explicit optional component.
- No automatic downloads during generation.
- Manual install/download from UI or installer.
- Clear status in settings: available / missing / failed.
- Runtime must be self-contained enough to avoid PATH issues.

Open questions:

- Can we reuse the Audacity OpenVINO DeepFilterNet3 assets locally?
- Is there a Windows-compatible DeepFilterNet3 runtime with native `libdf`
  available for Python 3.12?
- Should we package a small separate denoise executable instead of importing
  Python modules in the main server?
- Can OpenVINO Runtime be used directly with the DeepFilterNet3 model files?

## Provider Contract

Future code should expose a narrow provider interface:

```python
def is_available() -> bool:
    ...

def denoise_wav(input_path: Path, output_path: Path) -> DenoiseResult:
    ...
```

Suggested result fields:

```text
enabled
backend
input_path
output_path
duration_ms
sample_rate
status
error
metrics
```

This keeps SRT2Voice and voice creation independent from the chosen denoise
implementation.

## Rollback Rule

The denoiser must be removable without breaking Voicebox.

If the denoise provider is unavailable:

- generation still works
- SRT2Voice Auto Cut still works
- voice creation still works
- no model download is attempted automatically
- no exception reaches the user unless they explicitly requested denoise

## TODO

- Locate a Windows/Python 3.12 compatible DeepFilterNet3 runtime.
- Investigate whether the Audacity OpenVINO DeepFilterNet3 model can be reused.
- Add a safe denoise provider abstraction.
- Add optional SRT2Voice denoise status/debug.
- Add raw/denoised files to SRT2Voice export package.
- Add optional denoise preview for voice sample creation.
- Consider DNSMOS later as a QA layer, not as a denoise backend.
