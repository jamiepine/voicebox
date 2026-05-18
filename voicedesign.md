# VoiceDesign integration notes

## Validation status

Status on 2026-05-07: validated and functional.

- VoiceDesign profile creation works.
- The model is downloadable from the normal Models tab.
- Generation works from the main Voicebox voice generation flow.
- Generation also works from the Dubbing module.
- The integration keeps model download manual: generation must not trigger an
  implicit Hugging Face download.

VoiceDesign is implemented as a separate Qwen engine: `qwen_voice_design`.
It must not be treated as Base voice cloning or as CustomVoice presets.

## Engine contract

- Backend class: `backend/backends/qwen_voice_design_backend.py`
- Model repo: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- Model name in registry: `qwen-voice-design-1.7B`
- Engine id: `qwen_voice_design`
- Profile type: `designed`
- Profile payload must include `design_prompt`

## Technical integration

VoiceDesign is integrated as a first-class Voicebox engine, not as a Dubbing
special case. The goal is to keep it aligned with the existing Voicebox engine
contract:

```text
profile -> engine resolution -> model registry -> backend load -> generation
```

### Backend registry

`backend/backends/__init__.py`

- Adds `qwen_voice_design` to `TTS_ENGINES`.
- Adds `_get_qwen_voice_design_configs()`.
- Registers `qwen-voice-design-1.7B` in `get_all_model_configs()` and
  `get_tts_model_configs()`.
- Routes `qwen_voice_design` to `QwenVoiceDesignBackend` in
  `get_tts_backend_for_engine()`.
- Treats `qwen_voice_design` as a model-size engine so `1.7B` is preserved.
- Uses `ensure_model_cached_or_raise()` to prevent implicit model downloads
  during generation.

### Backend implementation

`backend/backends/qwen_voice_design_backend.py`

- Implements `QwenVoiceDesignBackend`.
- Uses the Qwen VoiceDesign repository:

```text
Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

- Exposes `_is_model_cached()` so the Models tab and generation preflight can
  detect local availability.
- Loads the model through `Qwen3TTSModel.from_pretrained()`.
- Generates speech with `generate_voice_design()`.
- Builds the effective instruction from:

```text
design_prompt + optional delivery instruction
```

The voice description remains the primary instruction. Delivery instructions
are appended only as a short generation-level directive.

### Profile contract

`backend/services/profiles.py`

- Adds `designed` voice profiles.
- Requires `design_prompt`.
- Forces designed profiles to use `default_engine = qwen_voice_design`.
- Rejects incompatible engines for designed profiles.
- Keeps cloned and preset profile behavior unchanged.

### Generation flow

`backend/routes/generations.py`
`backend/services/generation.py`

- Resolves the selected/profile default engine as usual.
- Validates profile/engine compatibility before creating a generation.
- Checks that the target model is already cached locally before queuing work.
- Runs through the same `run_generation()` and `generate_audio_sync()` paths as
  other engines.

Important rule:

```text
Generation must not download a model implicitly.
```

If `qwen-voice-design-1.7B` is missing, the API returns a clear error and the
user must download the model from the Models tab.

### Dubbing flow

`backend/services/dubbing.py`

- `resolve_dubbing_engine_for_profile()` maps designed profiles to
  `qwen_voice_design`.
- Dubbing uses the same generation service as other profile types.
- No VoiceDesign-specific dubbing heuristic should be added unless it is
  explicitly scoped to the Dubbing module.

Validated behavior:

```text
designed profile -> qwen_voice_design -> Qwen VoiceDesign 1.7B -> Dubbing segment generation
```

### Frontend profile creation

`app/src/components/VoiceProfiles/ProfileForm.tsx`
`app/src/stores/uiStore.ts`

- Adds a `Voice design` creation mode.
- Stores the user voice description as `designPrompt`.
- Sends it to the backend as `design_prompt`.
- Creates profiles with `voice_type = designed`.

### Frontend engine/model selection

`app/src/components/Generation/EngineModelSelector.tsx`
`app/src/components/Generation/FloatingGenerateBox.tsx`
`app/src/lib/hooks/useGenerationForm.ts`
`app/src/lib/constants/languages.ts`
`app/src/lib/api/types.ts`

- Adds `qwen_voice_design` to frontend engine types.
- Adds `Qwen VoiceDesign 1.7B` to engine selectors.
- Allows instruct/delivery text for VoiceDesign.
- Restricts designed profiles to the VoiceDesign engine.
- Blocks generation if `qwen-voice-design-1.7B` is not marked downloaded in
  `/models/status`.

### Models tab integration

`backend/backends/__init__.py`
`app/src/components/ServerSettings/ModelManagement.tsx`

- The backend exposes `qwen-voice-design-1.7B` from `/models/status`.
- The frontend includes `qwen-voice-design-*` in the Voice Generation section.
- The model is downloaded through the same manual model-download UI as the
  other engines.

Expected Models tab entry:

```text
Qwen VoiceDesign 1.7B
```

The backend loads the VoiceDesign checkpoint and calls:

```python
model.generate_voice_design(
    text=text,
    language=language,
    instruct=design_prompt,
)
```

If a generation-level delivery instruction is present, it is appended to the
voice design prompt as a short delivery directive. This is intentionally scoped
to the VoiceDesign backend.

## UI behavior

Voice creation exposes a third source:

- `Clone from audio`: existing cloned voice flow
- `Built-in voice`: existing preset flow, including Qwen CustomVoice
- `Voice design`: creates a `designed` profile with a natural-language prompt

Recommended French prompts:

```text
Voix masculine française naturelle, ton documentaire calme, accent parisien neutre.
```

```text
Voix féminine française naturelle, chaleureuse, articulation claire, ton pédagogique.
```

Keep prompts short and actor-like. Best target length is usually 10-40 words.
Avoid keyword spam and contradictory styles.

## Model download behavior

VoiceDesign must appear in the normal Models tab as:

```text
Qwen VoiceDesign 1.7B
qwen-voice-design-1.7B
```

Generation must not trigger an implicit Hugging Face download. If the model is
not local, the UI/backend must fail fast and ask the user to download it from
the Models tab. This keeps generation offline/predictable and matches the
manual-download behavior expected by Voicebox.

## Dubbing behavior

Dubbing accepts `designed` profiles and resolves them to `qwen_voice_design`.
This keeps Dubbing compatible with cloned, CustomVoice, and VoiceDesign profiles
without adding a Dubbing-specific hack.

The Dubbing rule still applies: changes must stay scoped to Dubbing behavior
unless they are part of the shared Voicebox engine/profile contract.

## Packaging notes

The PyInstaller server specs and `build_binary.py` must include:

```text
backend.backends.qwen_voice_design_backend
```

This avoids a backend import failure once the server is packaged.

The Windows build venv must keep NumPy compatible with Numba/qwen_tts. The
known-good pin used for the current build is:

```text
numpy==2.0.2
```

Do not upgrade NumPy past 2.0.x unless Numba/qwen_tts compatibility has been
verified first.

## Rollback

If VoiceDesign causes instability, rollback these files first:

- `backend/backends/qwen_voice_design_backend.py`
- `backend/backends/__init__.py`
- `backend/services/profiles.py`
- `backend/services/dubbing.py`
- `backend/models.py`
- `backend/build_binary.py`
- `backend/voicebox-server.spec`
- `backend/voicebox-server-cuda.spec`
- `app/src/components/VoiceProfiles/ProfileForm.tsx`
- `app/src/components/Generation/EngineModelSelector.tsx`
- `app/src/components/Generation/FloatingGenerateBox.tsx`
- `app/src/components/DubbingTab/DubbingTab.tsx`
- `app/src/lib/api/types.ts`
- `app/src/lib/constants/languages.ts`
- `app/src/lib/hooks/useGenerationForm.ts`
- `app/src/stores/uiStore.ts`

Do not change or overwrite the installed AppData backend/CUDA directory as part
of rollback. Rebuild from source, then install through the normal Voicebox flow.
