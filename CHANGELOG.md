<!-- This file is compiled automatically during the release workflow. -->
<!-- Do not edit manually — your changes will be overwritten. -->
<!-- To update the draft: ask the agent to use the draft-release-notes skill. -->
<!-- To finalize a release: ask the agent to use the release-bump skill. -->

# Changelog

## [Unreleased]

### Dictation and agents

- **Bring your own LLM for refinement and personalities.** Settings → Captures → Refinement gains a *Custom LLM endpoint (advanced)* section: point it at any server that speaks OpenAI's `/v1/chat/completions` (llama.cpp, vLLM, LM Studio, LocalAI, Ollama's OpenAI shim, OpenRouter, or OpenAI itself) with a model name and an optional API key, and dictation refinement and personality compose/rewrite go there instead of the bundled Qwen3. Leave the endpoint blank and nothing changes. The key is write-only: the settings API reports only whether one is stored. The dictation readiness checklist treats a configured endpoint as ready, the Models page keeps managing the on-device Qwen regardless, and a failing endpoint surfaces as a clear error naming the URL and status. — @ext-sakamoro ([#947](https://github.com/jamiepine/voicebox/pull/947))
### API

- **OpenAI-compatible text-to-speech API.** `POST /v1/audio/speech` and `GET /v1/models` accept OpenAI's request shape, so the official `openai` SDKs, Open WebUI, Home Assistant and any other OpenAI TTS client become a local, offline TTS by changing only the base URL. `voice` is a Voicebox profile (name or id, falling back to the per-client binding and default voice like `/speak`); `model` is a Voicebox engine or model variant, with `tts-1`, `tts-1-hd` and `gpt-4o-mini-tts` accepted as aliases for the profile's configured engine; `response_format` supports `wav`, `mp3`, `flac`, `opus`, `pcm` and (with ffmpeg) `aac`; `speed` is a pitch-preserving time stretch; `instructions` feeds the engine's instruct prompt. Errors use OpenAI's `{"error": {...}}` body, and the endpoint is documented on the MCP Server page. — @neuron-tech-ai (from [#656](https://github.com/jamiepine/voicebox/pull/656); fixes [#10](https://github.com/jamiepine/voicebox/issues/10))

## [0.6.0] - 2026-10-04

**The stability release.** 0.5 made Voicebox a dictation and agent-voice studio; 0.6 makes it hold up. Sixty-plus contributor pull requests landed since 0.5 and most of them fix something that used to go wrong in daily use: MLX crashed with `There is no Stream(gpu, 1) in current thread` when a model was loaded on one thread and used on another, memory grew with every generation and never came back on unload, a fully-cached model retried HuggingFace five times per file before it would load offline, the Docker image could not write its own data volume, dictation refused to arm without an LLM it was never going to call, and the backend test suite would not even collect. All of that is fixed. The headline features ride on top: native AMD ROCm on Windows and in Docker, five new interface languages, and Chatterbox multilingual running on Apple Silicon through MLX.

One change is breaking for existing MCP configs: the four MCP tools are renamed from dotted to underscore names (`voicebox.speak` → `voicebox_speak`, and so on) because Claude Desktop rejects dots in tool names. See the MCP section below.

### Stability and reliability

- **Every MLX op runs on one worker thread.** MLX's Metal stream registry is thread-local and `asyncio.to_thread` hands successive calls to different pool threads, so loading a model on one thread and generating on another raised `There is no Stream(gpu, 1) in current thread` and killed the request. All MLX load, generate, transcribe and Qwen3 LLM calls (TTS, STT, personality compose and rewrite, dictation refinement) now go through a single-worker executor, reload and inference are folded into one submission so an unload cannot land between them, and `unload_model()` stays off the worker so `/models/unload` cannot stall the event loop behind an in-flight generation. Reproduced and verified on an M2 Ultra; a regression test locks the single-thread guarantee without needing `mlx` installed. — @Rdbi-The-Ai, @shsunmoonlee, @davidstrouk ([#1154](https://github.com/jamiepine/voicebox/pull/1154), from [#989](https://github.com/jamiepine/voicebox/pull/989), [#909](https://github.com/jamiepine/voicebox/pull/909), [#789](https://github.com/jamiepine/voicebox/pull/789); fixes [#699](https://github.com/jamiepine/voicebox/issues/699))
- **MLX models release their memory on unload.** Unloading an MLX model dropped the Python reference but left the Metal allocator holding the weights, so switching models on Apple Silicon only ever grew. Unload now clears the prompt cache and the allocator, and a failing op's traceback can no longer pin the model binding alive. — @byevanli ([#1161](https://github.com/jamiepine/voicebox/pull/1161), from [#1074](https://github.com/jamiepine/voicebox/pull/1074))
- **No more memory growth across generations.** Consecutive TTS generations accumulated tensors and cache that were never freed. Post-generation cleanup now runs after every generation, including `/generate/stream`, and is best-effort so a cleanup failure cannot fail the request that just succeeded. — @devangkantharia ([#1149](https://github.com/jamiepine/voicebox/pull/1149), from [#1032](https://github.com/jamiepine/voicebox/pull/1032); fixes [#923](https://github.com/jamiepine/voicebox/issues/923))
- **TADA voice-prompt encode runs under `torch.inference_mode`**, so cloning no longer builds autograd graphs it never uses. — @fooSynaptic ([#955](https://github.com/jamiepine/voicebox/pull/955))
- **Runaway MLX Qwen generations are caught and retried.** A generation that keeps emitting audio chunks past its text is detected and retried instead of producing minutes of noise; when a retry cannot split the text further, the trimmed clip is kept rather than failing. — @AirHua-byte ([#964](https://github.com/jamiepine/voicebox/pull/964))
- **Kokoro trims trailing silence and run-on noise** on short prompts, trimming only the clip edges so inter-segment gaps are preserved. — @devangkantharia ([#1150](https://github.com/jamiepine/voicebox/pull/1150), from [#1030](https://github.com/jamiepine/voicebox/pull/1030); fixes [#960](https://github.com/jamiepine/voicebox/issues/960))
- **Cached models load offline without retrying HuggingFace.** Loading a fully-downloaded model now forces offline mode for the duration of the load, so it skips the network HEAD request (and its 5-retry backoff) for every config file instead of retrying each one in sequence before the app becomes ready. This reinstates the load-time `force_offline_if_cached` guard that 0.4.5 ([#530](https://github.com/jamiepine/voicebox/pull/530)) removed as a hotfix for the `_patch_mistral_regex` crash ([#526](https://github.com/jamiepine/voicebox/issues/526)); the wrapper installed in that release catches the crash at the source, so the guard no longer trips it. Because a load now fails hard offline when any file is missing, the Chatterbox, Chatterbox Turbo and TADA cache checks were extended to the small files their loaders also read (tokenizer files, `conds.pt`, TADA's Llama tokenizer mirror) so a snapshot missing one of them reports "not cached" and downloads online instead. — @alter ([#1130](https://github.com/jamiepine/voicebox/pull/1130), from [#1112](https://github.com/jamiepine/voicebox/pull/1112); fixes [#434](https://github.com/jamiepine/voicebox/issues/434))
- **Qwen3 LLM loads no longer flip the whole process offline.** The refinement model's loader set a process-global offline guard that then broke every other engine's downloads. — @dknoodle ([#924](https://github.com/jamiepine/voicebox/pull/924))
- **Download state is honest.** `/models/status` no longer reports an errored download as still in progress, and an orphaned `.incomplete` blob left by a killed download no longer counts as one either. — @dknoodle, @alealv ([#926](https://github.com/jamiepine/voicebox/pull/926), [#1133](https://github.com/jamiepine/voicebox/pull/1133) from [#967](https://github.com/jamiepine/voicebox/pull/967))
- **The server keeps serving after the app's stdout pipe closes.** When the desktop app's log pipe went away, every log write raised and took the request down with it; writes now go through a pipe-safe path. — @suardim ([#1141](https://github.com/jamiepine/voicebox/pull/1141), from [#1124](https://github.com/jamiepine/voicebox/pull/1124))
- **Deleting a voice deletes its generations instead of orphaning them**, the whole delete cascade commits once, a locked version file no longer aborts it, and the delete dialog says what goes. — @SEPURI-SAI-KRISHNA ([#1148](https://github.com/jamiepine/voicebox/pull/1148), from [#984](https://github.com/jamiepine/voicebox/pull/984))
- **SQLite runs in WAL mode with a 5 s busy timeout**, set from a connect hook so every connection gets it, and hot foreign keys and sort columns are indexed. — @neuron-tech-ai ([#1136](https://github.com/jamiepine/voicebox/pull/1136) from [#667](https://github.com/jamiepine/voicebox/pull/667), [#1139](https://github.com/jamiepine/voicebox/pull/1139) from [#666](https://github.com/jamiepine/voicebox/pull/666))
- **N+1 queries removed** from the history listing and from story item counts. — @youtsuhodev, @neuron-tech-ai ([#1143](https://github.com/jamiepine/voicebox/pull/1143) from [#1087](https://github.com/jamiepine/voicebox/pull/1087), [#663](https://github.com/jamiepine/voicebox/pull/663))
- **Requests honour what you asked for.** `VOICEBOX_FORCE_CPU` is respected in device selection and logged when it applies; `/models/load` and `/models/unload` act on the `model_name` you pass instead of always loading Qwen; a generation request that omits `engine` uses the profile's configured engine instead of silently defaulting to Qwen. — @ousamabenyounes, @SEPURI-SAI-KRISHNA, @nikhiljangid120 ([#1145](https://github.com/jamiepine/voicebox/pull/1145) from [#1064](https://github.com/jamiepine/voicebox/pull/1064), [#1138](https://github.com/jamiepine/voicebox/pull/1138) from [#988](https://github.com/jamiepine/voicebox/pull/988) fixing [#977](https://github.com/jamiepine/voicebox/issues/977), [#1140](https://github.com/jamiepine/voicebox/pull/1140) from [#1129](https://github.com/jamiepine/voicebox/pull/1129))
- **Smaller fixes**: audio for a failed generation returns 404 instead of 500 (@flowbrix, [#893](https://github.com/jamiepine/voicebox/pull/893)); avatar upload guards against a missing filename (@joj0hq, [#954](https://github.com/jamiepine/voicebox/pull/954)); CUDA library download is refused on platforms that have no CUDA assets instead of failing mid-way (@Bortlesboat, [#770](https://github.com/jamiepine/voicebox/pull/770)); the web API URL works for remote access (@vi-sch, [#550](https://github.com/jamiepine/voicebox/pull/550)); `datetime.utcnow()` is gone throughout (@neuron-tech-ai, [#1146](https://github.com/jamiepine/voicebox/pull/1146) from [#659](https://github.com/jamiepine/voicebox/pull/659)).

### Security

- **seroval upgraded to 1.5.3** (CVE-2026-59940), forced through bun overrides so `@tanstack/router-core` picks it up. — @anupamme ([#1132](https://github.com/jamiepine/voicebox/pull/1132), from [#985](https://github.com/jamiepine/voicebox/pull/985))

### Dictation and transcription

- **Dictation works without an LLM when auto-refine is off.** The readiness gate required the Qwen3 refinement model to be cached before the hotkey would arm, even when refinement was disabled and the model would never be called. The gate, the Captures checklist and the readiness polling now only consider the LLM when auto-refine is on, and toggling it updates the state immediately. — @devangkantharia ([#1147](https://github.com/jamiepine/voicebox/pull/1147), from [#1013](https://github.com/jamiepine/voicebox/pull/1013); fixes [#753](https://github.com/jamiepine/voicebox/issues/753))
- **Long-form Whisper transcription on the PyTorch path.** Clips over 30 s were truncated; language and task are now passed straight to Whisper's generate, long clips go through the long-form path, and clips under 30 s keep the pad-to-30s path. — @NoxDecima ([#1151](https://github.com/jamiepine/voicebox/pull/1151), from [#616](https://github.com/jamiepine/voicebox/pull/616))
- **Uploads are transcoded to WAV before STT**, which fixes WebM dictation in web mode, and the uploaded file's extension is kept so the decoder picks the right container. — @ivancorsetti, @e3o8o ([#957](https://github.com/jamiepine/voicebox/pull/957), [#903](https://github.com/jamiepine/voicebox/pull/903))
- **macOS paste lands in Electron apps.** The synthetic Cmd-V now sets the Command flag on the key-down event, so Slack, VS Code and other Electron apps accept the paste. — @kylebuxton ([#952](https://github.com/jamiepine/voicebox/pull/952))
- **The fn key can be part of a chord on macOS.** — @sridhar-3009 ([#950](https://github.com/jamiepine/voicebox/pull/950))
- **Linux**: the dictate pill no longer crashes at startup (the click-through toggle aborted on an unrealized GTK window), and system-audio capture selects the monitor device by name instead of setting `PULSE_SOURCE`. — @jboero, @sridhar-3009 ([#906](https://github.com/jamiepine/voicebox/pull/906), [#949](https://github.com/jamiepine/voicebox/pull/949))

### AMD ROCm

- **Native AMD GPU acceleration on Windows.** A ROCm backend with its own downloader, packaging (`scripts/package_rocm.py`) and Tauri launcher, shipped as release assets alongside the CUDA ones. — @JuanCMPDev ([#538](https://github.com/jamiepine/voicebox/pull/538); resolves [#531](https://github.com/jamiepine/voicebox/issues/531))
- **ROCm in Docker** via a compose overlay, with MIOpen stability env vars. — @JustInCache, @Xarianne ([#630](https://github.com/jamiepine/voicebox/pull/630), [#865](https://github.com/jamiepine/voicebox/pull/865))
- **ROCm setup on Linux.** Docker ROCm builds keep PyTorch on the ROCm wheel index during dependency installation so later installs do not replace it with CUDA wheels; the compose overlay joins the groups that own the GPU device nodes at startup instead of assuming Ubuntu render/video IDs; native Linux setup picks ROCm wheels for AMD GPUs and CUDA wheels for NVIDIA GPUs before installing. — @1337hero ([#817](https://github.com/jamiepine/voicebox/pull/817))
- **GPU architecture is detected before `HSA_OVERRIDE_GFX_VERSION` is set**, and an empty override is unset before torch loads instead of breaking it. — @singlaamitesh, @Xarianne ([#785](https://github.com/jamiepine/voicebox/pull/785), [#864](https://github.com/jamiepine/voicebox/pull/864))
- **Windows GPU detection** works again: the justfile recipe that read GPU information had a syntax error. — @xorss ([#669](https://github.com/jamiepine/voicebox/pull/669))

### Languages

- **Five new interface languages**: Spanish (@kmikodev, [#798](https://github.com/jamiepine/voicebox/pull/798)), French (@youtsuhodev, [#802](https://github.com/jamiepine/voicebox/pull/802)), Brazilian Portuguese (@Pedro31051, [#810](https://github.com/jamiepine/voicebox/pull/810)), Korean (@TedChang-Lim, [#814](https://github.com/jamiepine/voicebox/pull/814)) and Italian (@albanobattistella, [#904](https://github.com/jamiepine/voicebox/pull/904)). With English, Japanese, Simplified and Traditional Chinese, the app now ships in nine.
- **`voicebox_speak` speaks the profile's language** instead of forcing English. — @Zeptoniator ([#1137](https://github.com/jamiepine/voicebox/pull/1137), from [#1054](https://github.com/jamiepine/voicebox/pull/1054))
- **Four male Mandarin Kokoro voices** that were missing from the voice list. — @sidc124 ([#788](https://github.com/jamiepine/voicebox/pull/788))

### Chatterbox multilingual on Apple Silicon

- **The `chatterbox` engine runs on MLX on Apple Silicon.** It loads `mlx-community/chatterbox-multilingual-v3` through `mlx_audio` instead of the CPU-pinned PyTorch path, keeping zero-shot cloning and all 23 languages, on the shared MLX worker so it cannot hit the cross-thread Stream crash. Measured on an M2 Ultra: a 3.7 s English clip in 2.1 s (RTF 0.56) against 15 s on the PyTorch path, a 3.6 s French clip in 1.4 s against 16.2 s. The runaway-chunk retry covers it, and the S3 tokenizer repo it needs is tracked in model status and removed on delete. — @hassekf ([#1162](https://github.com/jamiepine/voicebox/pull/1162), from [#1039](https://github.com/jamiepine/voicebox/pull/1039))

### MCP

- **Breaking: tool names use underscores.** Claude Desktop rejects tool names containing dots, so the four tools are renamed: `voicebox.speak` → `voicebox_speak`, `voicebox.transcribe` → `voicebox_transcribe`, `voicebox.list_captures` → `voicebox_list_captures`, `voicebox.list_profiles` → `voicebox_list_profiles`. Agents and configs that call the dotted names must be updated; the HTTP endpoint, the stdio shim and the `X-Voicebox-Client-Id` binding are unchanged. — @jashwanth-reddy-g ([#1135](https://github.com/jamiepine/voicebox/pull/1135), from [#808](https://github.com/jamiepine/voicebox/pull/808); fixes [#790](https://github.com/jamiepine/voicebox/issues/790))
- **`voicebox_speak` takes a `model_size` parameter** to pick the Qwen model size per call. — @AhmedIrfan7 ([#895](https://github.com/jamiepine/voicebox/pull/895))
- **Ten missing paralinguistic delivery tags** added to `ParalinguisticInput`, with the active tag row kept in view and the full tag set documented. — @Margalit111 ([#1142](https://github.com/jamiepine/voicebox/pull/1142), from [#1113](https://github.com/jamiepine/voicebox/pull/1113))

### Docker

- **The runtime image works non-root under a mounted data volume.** The default compose left the `voicebox` user unable to write: the `./output` bind mount came up root-owned and the HuggingFace cache mount point did not exist in the image, so startup logged `Permission denied` and downloads and generation both failed. The image now pre-creates the cache and generations directories with the right owner, the entrypoint adopts the uid/gid of a host-owned bind mount instead of chowning host directories and fails fast with a clear message if a required directory is still unwritable, and `sox` is installed because qwen-tts's X-vector extractor shells out to it. — @LaurinceG4N0, @virtuallyunknown, @thiagomargatho, @boessu ([#1152](https://github.com/jamiepine/voicebox/pull/1152), from [#1084](https://github.com/jamiepine/voicebox/pull/1084), [#1017](https://github.com/jamiepine/voicebox/pull/1017), [#945](https://github.com/jamiepine/voicebox/pull/945))
- **No more SIGILL on CPUs without AVX-512.** `pedalboard` is pinned below 0.9.21, whose Linux wheels contain AVX-512 instructions that crash the import on older CPUs. `gcc`/`g++` are in the runtime image so Triton can compile its driver shim on NVIDIA hosts, and the `rocminfo` probe only runs when `/dev/kfd` exists, so CPU and NVIDIA containers stop logging a missing-AMD-GPU warning on every start. — @harryqt ([#1160](https://github.com/jamiepine/voicebox/pull/1160), from [#1114](https://github.com/jamiepine/voicebox/pull/1114))
- **The Docker image runs on Python 3.12.** Both build stages were on `python:3.11-slim` while the backend declares `>=3.12,<3.13`; they now use `python:3.12-slim`, matching the release workflow and the venv pin. ([#1166](https://github.com/jamiepine/voicebox/pull/1166))
- **Windows checkouts build**: CRLF line endings are normalised in the Dockerfile, and `.dockerignore` keeps the ROCm entrypoint. — @sridhar-3009, @shabeervpk ([#951](https://github.com/jamiepine/voicebox/pull/951), [#861](https://github.com/jamiepine/voicebox/pull/861))

### Interface

- **Long error toasts are readable and copyable.** The first line that fits is shown whole, the full original error text is preserved behind a copy button, and clipboard failures are handled. — @hakimio ([#1134](https://github.com/jamiepine/voicebox/pull/1134), from [#1058](https://github.com/jamiepine/voicebox/pull/1058))
- **Selects in the floating generate box open upward** so they are not clipped at the bottom of the window. — @andreolf ([#936](https://github.com/jamiepine/voicebox/pull/936); fixes [#928](https://github.com/jamiepine/voicebox/issues/928))
- **Naive-UTC timestamps are parsed consistently** in absolute dates. — @joj0hq ([#953](https://github.com/jamiepine/voicebox/pull/953))
- **Export filenames carry the generation id**, so two exports of the same text no longer overwrite each other. — @ivancorsetti ([#956](https://github.com/jamiepine/voicebox/pull/956))

### Developer experience

- **The backend test suite collects and runs again.** An `app` ↔ `routes` import cycle aborted collection, and `test_profile_duplicate_names` could not import; both are fixed. — @Zeptoniator, @Lvigentini ([#1131](https://github.com/jamiepine/voicebox/pull/1131), from [#1055](https://github.com/jamiepine/voicebox/pull/1055) and [#1011](https://github.com/jamiepine/voicebox/pull/1011))
- **`just setup` pins the Python venv to 3.12.** Setup preferred `python3.12` when it found one but silently fell back to 3.13 or bare `python3`, which then failed to install dependencies capped below 3.13 (kokoro, misaki, numpy, numba). Setup now requires 3.12, recreates a venv that is on the wrong version or whose pip doesn't run, falls through to the next candidate when one cannot build a working venv (Debian's separate `python3.12-venv`, Homebrew's `pyexpat` on macOS 26), and exits with the install command for your platform if none work. Windows resolves it through the `py -3.12` launcher with a uv fallback. `backend/pyproject.toml` now declares `requires-python = ">=3.12,<3.13"`. — @dhananjaypai08 ([#1144](https://github.com/jamiepine/voicebox/pull/1144), from [#1031](https://github.com/jamiepine/voicebox/pull/1031))
- **Python 3.13**: voice sample validation uses `audioop-lts` where the stdlib module is gone. — @maskedsyntax ([#853](https://github.com/jamiepine/voicebox/pull/853); fixes [#852](https://github.com/jamiepine/voicebox/issues/852))
- **`just setup` installs `mlx-lm` and `mlx-audio` on Apple Silicon** so a dev venv matches the release build. — @flowbrix ([#892](https://github.com/jamiepine/voicebox/pull/892))
- **The `voicebox-mcp` shim sidecar builds on Windows.** — @namuda ([#794](https://github.com/jamiepine/voicebox/pull/794))
- **Dev builds without full Xcode**: an `actool` failure is non-fatal for dev builds and stays fatal, with its full output, for release builds. — @neuron-tech-ai ([#1156](https://github.com/jamiepine/voicebox/pull/1156), from [#664](https://github.com/jamiepine/voicebox/pull/664))
- **macOS-only keyboard-layout symbols are gated behind `cfg`**, clearing dead-code warnings on Linux and Windows. — @youtsuhodev ([#831](https://github.com/jamiepine/voicebox/pull/831))
- **Docs**: the quick-start guide uses the current voice-profile terminology, and CONTRIBUTING names the right app identifier. — @alexgsummer, @jitendra205saini ([#963](https://github.com/jamiepine/voicebox/pull/963), [#863](https://github.com/jamiepine/voicebox/pull/863))

## [0.5.0] - 2026-04-22

**The Capture release.** Voicebox stops being just a voice-cloning studio and becomes a full AI voice studio. Hold a key anywhere on your machine, speak, release — the transcript lands in the focused text field. Flip the primitive around and any MCP-aware agent — Claude Code, Cursor, Spacebot — speaks back through an on-screen pill in one of your cloned voices. A local LLM sits between the two, so transcripts come out clean and voice profiles can carry a personality that reshapes what the agent says before it gets spoken.

### Dictation — speak anywhere, paste anywhere

- **Global hotkey capture.** Hold a customizable chord anywhere on your machine (defaults: right-Cmd + right-Option on macOS, right-Ctrl + right-Shift on Windows), speak, release. A floating on-screen pill walks through recording → transcribing → refining → done with a live elapsed timer. The transcript lands as clean text.
- **Push-to-talk and toggle modes, each with its own chord.** The default toggle chord adds Space to the push-to-talk chord. Holding PTT and tapping Space mid-hold upgrades a hold into a hands-free session without a gap in the recording.
- **Auto-paste into the focused app.** Once transcription finishes, Voicebox synthesizes a paste into whatever text field had focus when you started the chord — not wherever focus drifted while you were talking. Works across Dvorak / AZERTY layouts. Your clipboard is saved before and restored after.
- **Chord picker UI.** Customize either chord from Settings → Captures by holding the keys you want. Left/right modifier badges show whether a key is the left or right variant.
- **Defaults stay out of your way.** macOS defaults avoid left-hand Cmd+Option chords so the system shortcuts they collide with stay yours. Windows defaults route around AltGr collisions on German / French / Spanish layouts.
- **Accessibility permission is scoped.** If macOS Accessibility isn't granted, dictation still runs and transcripts still land in the Captures tab — only synthetic paste is disabled. The permission prompt lives inline next to the auto-paste toggle, not as a global banner.

### Personality — voice profiles that speak for themselves

Voice profiles now carry an optional **personality** — a free-form description of who this voice is, up to 2000 characters. When set, two new controls appear next to the generate button, each powered by a new Qwen3 LLM running entirely locally:

- **Compose** — the shuffle button drops a fresh in-character line into the textarea. Click again for variety, edit before speaking.
- **Speak in character** — the wand toggle runs your input through the personality LLM before TTS, preserving every idea but delivering it in the character's voice.

The same LLM doubles as the refinement model, so there's one local LLM in the app, not two.

**API surface.** `POST /generate`, `POST /speak`, and the MCP `voicebox.speak` tool accept `personality: bool`. `POST /profiles/{id}/compose` powers the shuffle button. MCP client bindings carry a `default_personality: bool` that applies when `personality` isn't passed explicitly.

### Agents — any MCP-aware agent gets a voice

Voicebox ships a built-in **Model Context Protocol** server at `http://127.0.0.1:17493/mcp` so Claude Code, Cursor, Windsurf, Cline, VS Code MCP extensions — any MCP-aware agent — can call into your local Voicebox install. Four tools ship with dotted names:

- **`voicebox.speak`** — speak text in any voice profile, with optional `personality: true` to run through the profile's personality LLM first
- **`voicebox.transcribe`** — Whisper transcription of a base64 blob or an absolute local path. Path mode is restricted to loopback callers so a Voicebox bound on `0.0.0.0` doesn't double as an unauthenticated arbitrary-local-file read primitive.
- **`voicebox.list_captures`** — recent captures with their transcripts
- **`voicebox.list_profiles`** — available voice profiles (cloned + preset)

- **Streamable HTTP as primary transport.** Cursor / Windsurf / VS Code / Claude Code all support it out of the box — drop a `mcpServers` block with the URL and an `X-Voicebox-Client-Id` header.
- **Stdio shim for clients that don't speak HTTP MCP.** A `voicebox-mcp` binary ships inside the app bundle as a Tauri sidecar. The Settings page renders the install snippet with the right absolute path pre-filled.
- **Per-client voice binding.** Pin Claude Code to Morgan, Cursor to Scarlett, Cline to its own voice — the `X-Voicebox-Client-Id` header resolves to a bound voice whenever `speak` is called without an explicit `profile`. Managed in **Settings → MCP**.
- **Profile resolution precedence.** Explicit `profile` arg (name or id, case-insensitive) → per-client binding → global default from `capture_settings.default_playback_voice_id` → error with a pointer to Settings.
- **Speaking pill.** Agent-initiated speech surfaces the same on-screen pill as dictation, in a `speaking` state with the profile name and an elapsed timer. Silent background TTS is a trust hazard — the pill always shows what's coming out of your machine.
- **`POST /speak` REST wrapper.** Same code path and voice resolution for shell scripts, ACP, A2A, GitHub Actions, or anything else that isn't MCP-native.

**Claude Code one-liner:**

```
claude mcp add voicebox --transport http --url http://127.0.0.1:17493/mcp --header "X-Voicebox-Client-Id: claude-code"
```

### Refinement

A clean transcript needs more than Whisper. Each capture flows through a small Qwen3 LLM that strips fillers, fixes punctuation, and optionally rewrites self-corrections — all on-device.

- **Loop-stripping before the LLM sees the transcript.** Whisper's "thanks for watching thanks for watching thanks for watching…" hallucination loops are collapsed at a six-identical-tokens threshold (case-insensitive) so a small refinement model can't echo them back. Coverage spans single-word runs, multi-word phrases, CJK character runs, and Japanese emphasis patterns; legitimate repetition ("no, no, no, no, no") doesn't cross the threshold.
- **Per-capture flag snapshot.** `smart_cleanup`, `self_correction`, and `preserve_technical` are stored on each capture, so refinement can be re-run later with different flags without losing the raw transcript.
- **Model picker** — Qwen3 0.6B (400 MB, very fast), 1.7B (1.1 GB, fast), 4B (2.5 GB, full quality). 0.6B is the default; 1.7B is the sweet spot for transcripts with code identifiers.

### Captures tab + settings

Settings → Captures is now the home for the whole dictation flow:

- **Dictation**: global shortcut toggle, push-to-talk chord picker, toggle chord picker, live pill preview, auto-paste into focused field (with inline accessibility prompt).
- **Transcription**: model picker (Whisper Base / Small / Medium / Large / Turbo), language lock.
- **Refinement**: auto-refine toggle, model picker, smart cleanup, remove self-corrections, preserve technical terms.
- **Playback**: default voice for the Captures tab's "Play as" action — picking a voice from the split-button persists the choice across tab switches and restarts.
- **Storage**: captures folder quick-open.

### Stories — timeline editor

The Stories tab graduates from a TTS sequencer into a real timeline editor. Same generation-row backing, but clips now compose with imported audio, per-clip levels, and a flexible track stack.

- **Import external audio.** Drag a music file onto the story content area or pick one from the new "Import audio" entry in the add-clip popover. Accepted formats: wav / mp3 / flac / ogg / m4a / aac / webm, capped at 200 MB. Imported clips show their filename instead of a profile name and skip the regenerate / version-picker controls — there's nothing to regenerate.
- **Per-clip volume.** A `Volume2` icon in the clip-edit toolbar opens a 0–200% slider. Adjustments apply live and to exports. Split and duplicate carry the volume forward into the new clips.
- **Regenerate** from both the clip's chat-list dropdown and the track-editor toolbar. Re-runs the underlying generation through the same path the History tab uses, with completion tracked in the global pending set.
- **Add empty tracks above or below the timeline** via tiny `+` strips at the top of the topmost label cell and the bottom of the bottommost. Sticky in the label column so they follow horizontal scroll.
- **Zoom bar tracks the project.** Min scope is 10 seconds visible (zoomed in cap), max is the entire project (zoomed out cap), default lands on 60 s. Both the +/− buttons and the scrollbar edge-drag handles clamp to those dynamic bounds.

### Interface

- **Theme selector.** Light / dark / system in **Settings → General**, persisted across sessions. System mode listens for OS-level appearance changes and flips live without a restart.
- **Scrubbable waveform player on captures.** The capture detail card now embeds a WaveSurfer waveform with click-to-seek and a current / total timestamp pair, replacing the static duration label.
- **Capture pill light mode.** The on-screen pill gets a dedicated light palette so it stays legible against bright windows.
- **Readiness checklist in the Captures settings sidebar.** The same six-gate checklist the Captures empty state uses mirrors into Settings → Captures so a red gate can't hide behind a green toggle. Hidden once every gate is green. macOS-only rows (Input Monitoring, Accessibility) hide entirely on Windows and Linux.

### Windows parity

Same dictation flow on Windows. Right-hand default chord (Ctrl+Shift) avoids AltGr collisions on layouts where Ctrl+Alt is the compose key. Focus is captured at chord-start so paste lands in the original field even if focus drifts during transcribe/refine.

## [0.4.5] - 2026-04-22

Second hotfix for the "offline mode is enabled" crash on model load. 0.4.4 reverted the inference-path offline guards but kept the same trap on the load path, so users who updated to 0.4.4 kept hitting the exact error the release was supposed to fix ([#526](https://github.com/jamiepine/voicebox/issues/526)). This release removes the load-path guards and patches the transformers tokenizer load to be robust to HuggingFace metadata failures at the source, so the class of bug can't recur.

### Reliability

- **Load no longer fails with "offline mode is enabled"** ([#530](https://github.com/jamiepine/voicebox/pull/530), fixes [#526](https://github.com/jamiepine/voicebox/issues/526)). transformers 4.57.x added an unconditional `huggingface_hub.model_info()` call inside `AutoTokenizer.from_pretrained` (via `_patch_mistral_regex`) that runs for every non-local repo load, regardless of cache state or whether the target model is actually a Mistral variant. The load-time `HF_HUB_OFFLINE` guard from 0.4.2 turned that into a hard crash for cached online users the moment 0.4.4 removed the inference-path guard that had been masking the problem. Fix wraps `_patch_mistral_regex` so any exception from the HF metadata check is caught and the tokenizer is returned unchanged — matching the success-path behavior for non-Mistral repos. The wrapper installs at `backend.backends` import time so it covers Qwen Base, Qwen CustomVoice, TADA, and every other transformers-backed engine on Windows, Linux, and CUDA alike. The load-time `force_offline_if_cached` guards were removed — with the wrapper in place they provide zero value and only risk re-introducing the same failure mode.
- **No more 30s pause when generating without a network.** The HuggingFace metadata timeout called out as a known caveat in 0.4.4 is covered by the same patch; offline users no longer wait for the check to time out before load completes.

## [0.4.4] - 2026-04-21

Hotfix for a regression in 0.4.3 where generation and transcription could fail outright with "offline mode is enabled" even when the user was online.

### Reliability

- **Inference no longer fails with "offline mode is enabled" while online** ([#524](https://github.com/jamiepine/voicebox/pull/524), reverts the inference-path guards from [#503](https://github.com/jamiepine/voicebox/pull/503)). 0.4.3 wrapped every inference body (`generate`, `transcribe`, `create_voice_clone_prompt`) with a process-wide `HF_HUB_OFFLINE` flip to stop lazy HuggingFace lookups from hanging when the network drops mid-inference ([#462](https://github.com/jamiepine/voicebox/issues/462)). That flag also blocks legitimate metadata calls (e.g. `HfApi().model_info` for revision resolution) so online users started seeing generation fail outright. Inference now runs with the process's default HF state. Load-time offline guards — which weren't the source of the regression — stay in place.

**Known caveat**: users generating without an internet connection may see brief pauses during inference while HuggingFace metadata lookups time out (typically ~30s, after which the library recovers). A proper offline-mode toggle is planned for 0.4.5.

## [0.4.3] - 2026-04-20

A patch focused on two user-impacting reliability fixes: macOS DMG notarization (unblocks `brew install voicebox` on macOS 15 Sequoia and fixes spurious "app isn't signed" Gatekeeper dialogs on older Intel Macs) and Kokoro Japanese voice initialization on fresh installs.

### macOS

- **DMGs are now notarized and stapled** ([#523](https://github.com/jamiepine/voicebox/pull/523)). Tauri's bundler notarizes the `.app` inside the DMG but ships the DMG wrapper itself unnotarized. Gatekeeper rejects that on macOS 15 Sequoia (confirmed by Homebrew Cask CI failing on both arm and intel Sequoia runners) and causes the "the app is not signed" dialog on older Intel Macs when Apple's notarization servers are slow or unreachable ([#509](https://github.com/jamiepine/voicebox/issues/509)). The release workflow now submits each DMG to `notarytool`, staples the ticket, verifies with `spctl`, and overwrites the draft-release asset `tauri-action` uploaded. Adds ~5-10 min per macOS job.

### Backend

- **Kokoro Japanese voices no longer crash on fresh installs** ([#521](https://github.com/jamiepine/voicebox/pull/521), fixes [#514](https://github.com/jamiepine/voicebox/issues/514)). `misaki[ja]` pulls in `fugashi`, which needs a MeCab dictionary on disk. The `unidic` package that was being installed ships no data and expects a ~526MB runtime download that `just setup` doesn't run (and which wouldn't survive PyInstaller anyway). Swapped to `unidic-lite`, which bundles a MeCab-compatible dict inside the wheel (~50MB). Collected in `build_binary.py` so frozen builds pick up `unidic_lite/dicdir/`.

## [0.4.2] - 2026-04-20

This release localizes the entire app. English, Simplified Chinese (zh-CN), Traditional Chinese (zh-TW), and Japanese (ja) are wired up end-to-end across every tab, modal, dialog, and toast — 559 translation keys per locale, parity verified. Plus a batch of reliability fixes: offline-mode now actually stays offline, Chatterbox accepts reference samples it used to reject, MLX Qwen 0.6B points at the right repo, and macOS system audio survives backgrounding.

### Internationalization ([#508](https://github.com/jamiepine/voicebox/pull/508))
- **i18next foundation** with an in-app language switcher that re-renders the tree on change — lazy-loaded components were holding stale strings without an explicit key-bump on the React root.
- **Four locales** at full coverage: English, Simplified Chinese, Traditional Chinese, Japanese. No partial/English-fallback surfaces.
- **Every user-visible surface translated**: Stories (list, content editor, dialogs, toasts), Effects (list, detail, chain editor, built-in preset names), Voices (table, search, inspector, Create/Edit modal, audio sample panels), Audio Channels (list, dialogs, device picker), history + story dropdown menus, ProfileCard / ProfileList / HistoryTable, and the unsupported-model note.
- **Relative dates** localize via `date-fns` locale objects (`3 days ago` → `3 天前` / `3 日前`) — `Intl.RelativeTimeFormat` doesn't produce the phrasing we use in the history table.
- **Dev-build version suffix** (`v0.4.2 (dev)` / `(开发版)` / `(開發版)` / `(開発版)`) is now locale-aware.
- **559 translation keys** across all four locales.

### Reliability
- **`HF_HUB_OFFLINE` now guards every inference path** ([#503](https://github.com/jamiepine/voicebox/pull/503)) — some engines were still attempting a HuggingFace metadata roundtrip on first load when offline mode was enabled, causing hangs on airgapped or flaky networks.
- **Chatterbox reference samples are preprocessed instead of rejected** ([#502](https://github.com/jamiepine/voicebox/pull/502)) — samples outside the expected sample rate or channel layout are resampled to match, rather than failing with an opaque error.
- **MLX Qwen 0.6B repo path fixed** ([#501](https://github.com/jamiepine/voicebox/pull/501)) — now points at the published `mlx-community` repo so the model actually downloads on Apple Silicon.
- **macOS system audio survives backgrounding** ([#486](https://github.com/jamiepine/voicebox/pull/486), closes [#41](https://github.com/jamiepine/voicebox/issues/41)) — WKWebView was tearing down the audio session when the app lost focus, silently killing system-audio capture.
- **MLX backend `miniaudio` dependency pinned** ([#506](https://github.com/jamiepine/voicebox/pull/506)) — `mlx_audio.stt` needs it at runtime and nothing else transitively pulled it in, so `--no-deps` installs were breaking on first use.

### Landing / Docs
- **New `/download` page** ([#487](https://github.com/jamiepine/voicebox/pull/487)) — no more dumping first-time visitors onto the GitHub releases list. The API example snippet on the landing page also got an accuracy pass.
- **Download redirects work behind reverse proxies** ([#498](https://github.com/jamiepine/voicebox/pull/498)) — uses the public origin instead of `localhost` when resolving platform-specific installer URLs.
- **MDX docs audited against the multi-engine backend** ([#484](https://github.com/jamiepine/voicebox/pull/484)) — stale single-engine assumptions removed.
- **Three more tutorials + mobile navbar / hero CTA fixes** ([#483](https://github.com/jamiepine/voicebox/pull/483)).

### Linux
- **Still not shipping.** The re-enable attempt ([#488](https://github.com/jamiepine/voicebox/pull/488)) landed on `main` but CI still hangs in the `tauri-action` bundler step on `ubuntu-22.04` — no output for 25+ minutes after `rpm` bundling, even with `createUpdaterArtifacts: false` and `--bundles deb,rpm`. The matrix entry is disabled again for 0.4.2; the ubuntu-specific setup steps stay in the workflow so re-enabling is a one-line change once we identify the hang. Next release will take another pass.

### New Contributors
- [@shekharyv](https://github.com/shekharyv) — download redirects behind reverse proxies ([#498](https://github.com/jamiepine/voicebox/pull/498))

## [0.4.1] - 2026-04-18

A fast follow-up to 0.4.0 focused on making the new engines actually load in the production binary — plus generation cancellation, Linux system-audio capture, and the repo's first PR-time type check. Five first-time contributors shipped in this release.

0.4.0 introduced three new TTS engines, but the frozen PyInstaller binary tripped over several Python-ecosystem quirks that don't show up in the dev venv: `transformers` opening `.py` sources at runtime, `scipy.stats._distn_infrastructure` hitting a frozen-importer `NameError`, and `chatterbox-multilingual` failing to find its Chinese segmenter dictionary. This release patches all of those in one sweep.

### Frozen-Binary Reliability ([#438](https://github.com/jamiepine/voicebox/pull/438))
- **Kokoro** now bundles `.py` sources alongside `.pyc` via `--collect-all kokoro` so `transformers`' `_can_set_attn_implementation` regex scan can read them — previously `FileNotFoundError: kokoro/modules.py` killed Kokoro loading in production builds
- **Chatterbox Multilingual** now bundles `spacy_pkuseg/dicts/default.pkl` and the package's native `.so` extensions via `--collect-all spacy_pkuseg` — previously the Chinese word segmenter crashed with `FileNotFoundError` on first load
- **scipy.stats._distn_infrastructure** — new runtime hook source-patches the trailing `del obj` (which raises `NameError` under PyInstaller's frozen importer because the preceding list comprehension evaluates empty) to `globals().pop('obj', None)`, unblocking `librosa` → `scipy.signal` → `scipy.stats` for every TTS engine that depends on librosa
- **transformers.masking_utils** — same runtime hook forces `_is_torch_greater_or_equal_than_2_6 = False` so the older `sdpa_mask_older_torch` path is selected; the 2.6+ path uses `TransformGetItemToIndex()`, a real `torch._dynamo` graph transform our permissive stub can't reproduce
- **torch._dynamo** — no-op stub replaces the real module before `transformers` imports it, preventing the `torch._numpy._ufuncs` import crash (`NameError: name 'name' is not defined`) that blocked Kokoro and every engine pulling in `flex_attention`
- `.spec` paths are now repo-relative instead of absolute, so the generated spec is portable across machines and CI

### Generation
- **Cancel queued or running generations** ([#444](https://github.com/jamiepine/voicebox/pull/444)) — new `/generate/{id}/cancel` endpoint and a Stop button on the history row while generating. The serial queue now tracks per-ID state (queued / running / cancelled) so queued jobs are skipped before the worker picks them up and running jobs are `.cancel()`-ed mid-flight; `run_generation` catches `CancelledError` and marks the row `failed` with a "cancelled" error.
- **Legacy `data/` path prefix resolution** ([#440](https://github.com/jamiepine/voicebox/pull/440)) — generations stored with the old `data/` prefix under pre-0.4 installs now resolve correctly after the storage root moved, fixing 404s for historical audio.

### Model Migration
- Migration dialog no longer hangs when the cache is empty ([#439](https://github.com/jamiepine/voicebox/pull/439)) — the backend now emits a completion SSE event even when zero models are moved.
- Storage-change flow surfaces a toast when there's nothing to migrate ([#433](https://github.com/jamiepine/voicebox/pull/433)) instead of proceeding with a no-op move and restarting the server.
- Deleting all generations from a voice profile now deletes the associated version files and DB rows too ([#447](https://github.com/jamiepine/voicebox/pull/447)) — previously orphaned versions accumulated in storage.

### Platform
- **Linux system audio capture** ([#457](https://github.com/jamiepine/voicebox/pull/457)) — `cpal`'s ALSA backend doesn't expose PulseAudio/PipeWire monitor sources by name, so the previous device-name search never matched and silently fell back to the microphone. Detection now uses `pactl get-default-sink` + `pactl list short sources` and routes via `PULSE_SOURCE`, with the name-based search retained as a fallback when `pactl` is absent.

### Frontend CI
- First PR-time quality gate ([#418](https://github.com/jamiepine/voicebox/pull/418)) — new `.github/workflows/ci.yml` runs `bun run typecheck` + `bun run build:web` on every PR. Fixed pre-existing type issues that were being suppressed with `@ts-expect-error`, cleaned up a dep-array typo (`[platform.metadata.isTauricheckOnMountcheckForUpdates]`) in `useAutoUpdater`, and removed 100+ lines of dead `ModelItem` code from `ModelManagement.tsx`.
- Follow-up: widened `apiClient.migrateModels()` return type to include `moved` and `errors` so the storage-change handler typechecks against the real backend response ([#470](https://github.com/jamiepine/voicebox/pull/470)).

### Docs
- Clarified in the Quick Start + README that paralinguistic tags (`[laugh]`, `[sigh]`) only work with Chatterbox Turbo; other engines read them as literal text ([#450](https://github.com/jamiepine/voicebox/pull/450)).

### New Contributors
- [@Bortlesboat](https://github.com/Bortlesboat) — generation cancellation (#444)
- [@gaojulong](https://github.com/gaojulong) — migration dialog hang fix (#439)
- [@fuleinist](https://github.com/fuleinist) — migration no-op toast (#433)
- [@erionjuniordeandrade-a11y](https://github.com/erionjuniordeandrade-a11y) — frontend CI + type hardening (#418)
- [@estefrac](https://github.com/estefrac) — Linux pactl system-audio capture (#457)

## [0.4.0] - 2026-04-16

The biggest Voicebox release yet. Three new TTS engines bring the lineup to **seven** — HumeAI TADA, Kokoro 82M, and Qwen CustomVoice join Qwen3-TTS, LuxTTS, Chatterbox Multilingual, and Chatterbox Turbo. GPU support broadens to Intel Arc (XPU) and NVIDIA Blackwell (RTX 50-series), with runtime diagnostics that warn when your PyTorch build doesn't match your GPU. The CUDA backend is now split into independently versioned server and library archives, so upgrading no longer redownloads 4 GB of PyTorch/CUDA DLLs.

This release also marks a big community moment: **13 new contributors** shipped fixes and features in 0.4.0. Thirty-plus bug fixes target the most-reported issues in the tracker — numpy 2.x TTS crashes, Windows background-server reliability, macOS 11 launch failures, audio playback silence, Stories clip-splitting races, history status staleness, and more.

### New TTS Engines

#### HumeAI TADA — Expressive English & Multilingual ([#296](https://github.com/jamiepine/voicebox/pull/296))
- Added `tada-1b` (English) and `tada-3b-ml` (multilingual) backends
- Replaced `descript-audio-codec` with a lightweight DAC shim to cut dependencies
- Switched audio decoding to `soundfile` to sidestep `torchcodec` bundling issues
- Redirected gated Llama tokenizer lookups to an ungated mirror so model loading works out of the box
- Fixed tokenizer patch that was corrupting `AutoTokenizer` for other engines
- Fixed TorchScript error in frozen builds

#### Kokoro 82M — Fast Lightweight TTS ([#325](https://github.com/jamiepine/voicebox/pull/325))
- Added Kokoro 82M engine with a new voice profile type system that distinguishes preset voices from cloned profiles
- Profile grid now handles engine compatibility directly — removed redundant dropdown filtering
- Tightened Kokoro profile handling so preset voices can't be edited like cloned profiles

#### Qwen CustomVoice ([#328](https://github.com/jamiepine/voicebox/pull/328))
- Added `qwen-custom-voice` preset engine backed by Qwen3-TTS
- Enforced preset/profile engine compatibility across the generation flow
- Floating generator now shows all engines instead of silently filtering

### Voice Profile UX

Until 0.4, every engine in Voicebox was a cloning model, so every voice profile was usable with every engine and the profile grid just showed them all. Introducing Kokoro and Qwen CustomVoice — which work from preset voices rather than cloned samples — broke that assumption for the first time. An early cut on `main` filtered the grid by the selected engine, which left users running pre-release builds thinking their cloned voices had vanished whenever they switched to a preset-only engine.

This release ships the resolution before it ever reaches a tagged version:

- **Grey-out instead of filter** — all profiles are always visible; unsupported ones render dimmed with a compatibility hint at the bottom of the grid
- **Auto-switch on selection** — clicking a greyed-out profile selects it AND switches the engine to a compatible one, instead of silently doing nothing
- **Instruct toggle restored for Qwen CustomVoice** — the floating generate box now reveals a delivery-instructions input (tone, emotion, pace) when CustomVoice is selected. Hidden across the board while the new multi-engine lineup was stabilizing because most engines don't honor the kwarg; now conditionally exposed only for the one engine that was actually trained for instruction-based style control
- Supported profiles sort first; the grid scrolls the selected profile into view after engine/sort changes
- Fixed engine desync on tab navigation — the form now initializes its engine from the store
- Fixed the disabled-and-selected card click edge case by bouncing selection to re-trigger the auto-switch
- Cleaned up scroll effect timers (requestAnimationFrame + setTimeout) to prevent stale DOM writes on unmount or rapid selection changes

### GPU & Platform

#### Intel Arc (XPU) Support ([#320](https://github.com/jamiepine/voicebox/pull/320))
- First-class Intel Arc support across all PyTorch-based backends
- Device-aware seeding, XPU detection in the GPU status panel, and setup flow detection
- Reports correct device name and VRAM in settings

#### Blackwell / RTX 50-series Support ([#316](https://github.com/jamiepine/voicebox/pull/316), [#401](https://github.com/jamiepine/voicebox/pull/401))
- Upgraded the CUDA backend from cu126 → cu128 for RTX 50-series support
- Added `sm_120+PTX` to the CUDA build via `TORCH_CUDA_ARCH_LIST` for forward-compatibility with Blackwell architectures (closes 5 open reports: #386, #395, #396, #399, #400)
- GPU settings UI fixes around install/uninstall state

#### GPU Compatibility Diagnostics ([#367](https://github.com/jamiepine/voicebox/pull/367), adapted)
- New `check_cuda_compatibility()` compares the current device's compute capability against the bundled PyTorch's architecture list
- Health endpoint exposes a `gpu_compatibility_warning` field so the UI can surface mismatches
- Startup logs a `WARN` when the installed PyTorch build doesn't support the detected GPU
- GPU status label shows `[UNSUPPORTED - see logs]` — no more silent "no kernel image" failures

#### Split CUDA Backend ([#298](https://github.com/jamiepine/voicebox/pull/298))
- CUDA backend now ships as two independently versioned archives: a small server binary and a large libs archive (the ~4 GB of PyTorch/CUDA DLLs)
- Upgrading Voicebox no longer redownloads the libs archive when only the server binary changed
- Added `asyncio.Lock` around `download_cuda_binary()` so auto-update and manual download can't race on the same temp file ([#428](https://github.com/jamiepine/voicebox/pull/428))
- Updated `package_cuda.py` for PyInstaller 6.18 onedir layout
- Temp archives are always cleaned up on failure, even when the install aborts mid-extract

### Bug Fixes

#### Critical: TTS Generation
- **numpy 2.x `torch.from_numpy` crash** ([#361](https://github.com/jamiepine/voicebox/pull/361)) — torch compiled against numpy 1.x ABI fails silently when paired with numpy 2.x, causing `RuntimeError: Numpy is not available` / `Unable to create tensor` on every TTS request in bundled macOS Intel / Rosetta builds. Pinned `numpy<2.0` in requirements and added a PyInstaller runtime hook with a `ctypes.memmove` fallback as belt-and-suspenders. Hardened afterward to raise on unknown dtypes instead of silently reinterpreting bytes as float32.

#### Platform Reliability
- **Windows background server** ([#402](https://github.com/jamiepine/voicebox/pull/402)) — "keep server running after close" now actually keeps the server running. The HTTP `/watchdog/disable` request could lose the race against process exit on Windows; added a `.keep-running` sentinel file as a synchronous fallback, with stale-sentinel cleanup on startup to avoid orphan server processes
- **macOS 11 launch crash** ([#424](https://github.com/jamiepine/voicebox/pull/424)) — weak-linked ScreenCaptureKit so the app can launch on macOS < 12.3 instead of crashing at dyld resolution. Gated system audio capture behind a real `sw_vers` version check so unsupported systems cleanly advertise "not available" rather than crashing at runtime
- **macOS Intel (x86_64) setup** ([#416](https://github.com/jamiepine/voicebox/pull/416)) — relaxed `torch>=2.7.0` → `torch>=2.2.0`. PyTorch dropped pre-built x86_64 wheels after 2.2.2, so Intel Mac devs could no longer `pip install`. Now resolves to the latest compatible torch per platform
- **Offline model loading** ([#318](https://github.com/jamiepine/voicebox/pull/318)) — Qwen TTS and Whisper force offline mode when loading cached models, so startup works without network access
- **GUI startup with external server** ([#319](https://github.com/jamiepine/voicebox/pull/319)) — fixed GUI launch when pointed at a remote/external server, and added data refresh on server switch; hardened health validation and error handling
- **Qwen3-TTS cache split on Windows** (adapted from [#218](https://github.com/jamiepine/voicebox/pull/218)) — route `Qwen3TTSModel.from_pretrained` through `hf_constants.HF_HUB_CACHE` so the speech tokenizer and `preprocessor_config.json` resolve from a single cache root
- **Qwen3-TTS bundling** ([#305](https://github.com/jamiepine/voicebox/pull/305)) — bundle `qwen_tts` source files in the PyInstaller build to fix `inspect.getsource` errors in frozen builds
- **Backend import paths** ([#345](https://github.com/jamiepine/voicebox/pull/345)) — moved lazy imports to top-level with absolute paths to resolve the "Failed to Save" preset error caused by `ModuleNotFoundError` in production builds
- **Effects service import** ([#384](https://github.com/jamiepine/voicebox/pull/384)) — fixed `ModuleNotFoundError` on preset create/update by switching to relative imports (#349)

#### Audio & Playback
- **cpal stream silent playback** ([#405](https://github.com/jamiepine/voicebox/pull/405)) — `cpal::Stream` was dropped on function return immediately after `play()`, causing every playback to fall silent. Now holds the stream until either the buffer drains or the stop flag fires (#404)

#### Stories & History
- **Clip-splitting race** ([#403](https://github.com/jamiepine/voicebox/pull/403)) — rapid double-clicks on split could race through `split_story_item` with inconsistent state. Added `with_for_update()` row locking on the backend and an `isPending` guard on the frontend (#366)
- **History `status` staleness** ([#394](https://github.com/jamiepine/voicebox/pull/394)) — `GET /history/{id}` was hardcoding `status="completed"` regardless of the DB row, breaking any client polling for job completion. Now returns `status`, `error`, `engine`, `model_size`, and `is_favorited` from the actual row
- **"Clear failed" bulk button** ([#412](https://github.com/jamiepine/voicebox/pull/412)) — new `DELETE /history/failed` endpoint and a header strip showing `"N failed generations"` with a Clear button, complementing the per-row trash icon added in #321 (#410)
- **Delete failed generations** ([#321](https://github.com/jamiepine/voicebox/pull/321)) — added a trash icon next to the retry button so failed entries can be cleaned up without having to retry first

#### Security & Safety
- **Voice prompt cache hardening** ([#429](https://github.com/jamiepine/voicebox/pull/429)) — `torch.load(weights_only=True)` on cached voice prompts per PyTorch 2.6 recommendation; replaced string-based SPA path guard with `Path.is_relative_to()` for more robust path-traversal protection

#### Infrastructure & Docker
- **Docker web build** ([#344](https://github.com/jamiepine/voicebox/pull/344)) — include `CHANGELOG.md` in the Docker web build so the in-app changelog page works in Docker deployments
- **Docker numba cache** ([#425](https://github.com/jamiepine/voicebox/pull/425)) — set `NUMBA_CACHE_DIR` in docker-compose so numba can write its JIT cache in container runtime (#308)
- **Relative media paths** ([#332](https://github.com/jamiepine/voicebox/pull/332)) — media paths now stored relative to the configured data dir rather than resolved against CWD, so the data directory is portable between installs

### Developer Tooling

- New `triage-prs` agent skill — encodes the end-to-end PR-speedrun workflow (classification → triage doc → rebase → squash-merge → follow-ups) so future release cycles can reproduce it
- Rewrote the TTS engine guide with the patterns learned from adding TADA and Kokoro
- Added the API refactor plan and CUDA libs addon design doc
- Fixed broken links in the Get Started section ([#332](https://github.com/jamiepine/voicebox/pull/332))

### New Contributors

Huge thank you to everyone who contributed their first PR to Voicebox in this release:

[@liorshahverdi](https://github.com/liorshahverdi), [@nicoschtein](https://github.com/nicoschtein), [@ArfianID](https://github.com/ArfianID), [@aimaaaimaa](https://github.com/aimaaaimaa), [@maxmcoding](https://github.com/maxmcoding), [@Khalodddd](https://github.com/Khalodddd), [@LuisSambrano](https://github.com/LuisSambrano), [@shaun0927](https://github.com/shaun0927), [@malletfils](https://github.com/malletfils), [@mvanhorn](https://github.com/mvanhorn), [@kuishou68](https://github.com/kuishou68), [@txhno](https://github.com/txhno), [@MukundaKatta](https://github.com/MukundaKatta)

## [0.3.0] - 2026-03-17

This release rewrites the backend into a modular architecture, overhauls the settings UI into routed sub-pages, fixes audio player freezing, migrates documentation to Fumadocs, and ships a batch of bug fixes targeting the most-reported issues from the tracker.

The backend's 3,000-line monolith `main.py` has been decomposed into domain routers, a services layer, and a proper database package. A style guide and ruff configuration now enforce consistency. On the frontend, settings have been split into dedicated routed pages with server logs, a changelog viewer, and an about page. The audio player no longer freezes mid-playback, and model loading status is now visible in the UI. Seven user-reported bugs have been fixed, including server crashes during sample uploads, generation list staleness, cryptic error messages, and CUDA support for RTX 50-series GPUs.

### Settings Overhaul ([#294](https://github.com/jamiepine/voicebox/pull/294))
- Split settings into routed sub-tabs: General, Generation, GPU, Logs, Changelog, About
- Added live server log viewer with auto-scroll
- Added in-app changelog page that parses `CHANGELOG.md` at build time
- Added About page with version info, license, and generation folder quick-open
- Extracted reusable `SettingRow` component for consistent setting layouts

### Audio Player Fix ([#293](https://github.com/jamiepine/voicebox/pull/293))
- Fixed audio player freezing during playback
- Improved playback UX with better state management and listener cleanup
- Fixed restart race condition during regeneration
- Added stable keys for audio element re-rendering
- Improved accessibility across player controls

### Backend Refactor ([#285](https://github.com/jamiepine/voicebox/pull/285))
- Extracted all routes from `main.py` into 13 domain routers under `backend/routes/` — `main.py` dropped from ~3,100 lines to ~10
- Moved CRUD and service modules into `backend/services/`, platform detection into `backend/utils/`
- Split monolithic `database.py` into a `database/` package with separate `models`, `session`, `migrations`, and `seed` modules
- Added `backend/STYLE_GUIDE.md` and `pyproject.toml` with ruff linting config
- Removed dead code: unused `_get_cuda_dll_excludes`, stale `studio.py`, `example_usage.py`, old `Makefile`
- Deduplicated shared logic across TTS backends into `backends/base.py`
- Improved startup logging with version, platform, data directory, and database stats
- Fixed startup database session leak — sessions now rollback and close in `finally` block
- Isolated shutdown unload calls so one backend failure doesn't block the others
- Handled null duration in `story_items` migration
- Reject model migration when target is a subdirectory of source cache

### Documentation Rewrite ([#288](https://github.com/jamiepine/voicebox/pull/288))
- Migrated docs site from Mintlify to Fumadocs (Next.js-based)
- Rewrote introduction and root page with content from README
- Added "Edit on GitHub" links and last-updated timestamps on all pages
- Generated OpenAPI spec and auto-generated API reference pages
- Removed stale planning docs (`CUDA_BACKEND_SWAP`, `EXTERNAL_PROVIDERS`, `MLX_AUDIO`, `TTS_PROVIDER_ARCHITECTURE`, etc.)
- Sidebar groups now expand by default; root redirects to `/docs`
- Added OG image metadata and `/og` preview page

### UI & Frontend
- Added model loading status indicator and effects preset dropdown ([3187344](https://github.com/jamiepine/voicebox/commit/3187344))
- Fixed take-label race condition during regeneration
- Added accessible focus styling to select component
- Softened select focus indicator opacity
- Addressed 4 critical and 12 major issues from CodeRabbit review

### Bug Fixes ([#295](https://github.com/jamiepine/voicebox/pull/295))
- Fixed sample uploads crashing the server — audio decoding now runs in a thread pool instead of blocking the async event loop ([#278](https://github.com/jamiepine/voicebox/issues/278))
- Fixed generation list not updating when a generation completes — switched to `refetchQueries` for reliable cache busting, added SSE error fallback, and page reset on completion ([#231](https://github.com/jamiepine/voicebox/issues/231))
- Fixed error toasts showing `[object Object]` instead of the actual error message ([#290](https://github.com/jamiepine/voicebox/issues/290))
- Added Whisper model selection (`base`, `small`, `medium`, `large`, `turbo`) and expanded language support to the `/transcribe` endpoint ([#233](https://github.com/jamiepine/voicebox/issues/233))
- Upgraded CUDA backend build from cu121 to cu126 for RTX 50-series (Blackwell) GPU support ([#289](https://github.com/jamiepine/voicebox/issues/289))
- Handled client disconnects in SSE and streaming endpoints to suppress `[Errno 32] Broken Pipe` errors ([#248](https://github.com/jamiepine/voicebox/issues/248))
- Fixed Docker build failure from pip hash mismatch on Qwen3-TTS dependencies ([#286](https://github.com/jamiepine/voicebox/issues/286))
- Added 50 MB upload size limit with chunked reads to prevent unbounded memory allocation on sample uploads
- Eliminated redundant double audio decode in sample processing pipeline

### Platform Fixes
- Replaced `netstat` with `TcpStream` + PowerShell for Windows port detection ([#277](https://github.com/jamiepine/voicebox/pull/277))
- Fixed Docker frontend build and cleaned up Docker docs
- Fixed macOS download links to use `.dmg` instead of `.app.tar.gz`
- Added dynamic download redirect routes to landing site

### Release Tooling
- Added `draft-release-notes` and `release-bump` agent skills
- Wired CI release workflow to extract notes from `CHANGELOG.md` for GitHub Releases
- Backfilled changelog with all historical releases

## [0.2.3] - 2026-03-15

The "it works in dev but not in prod" release. This version fixes a series of PyInstaller bundling issues that prevented model downloading, loading, generation, and progress tracking from working in production builds.

### Model Downloads Now Actually Work

The v0.2.1/v0.2.2 builds could not download or load models that weren't already cached from a dev install. This release fixes the entire chain:

- **Chatterbox, Chatterbox Turbo, and LuxTTS** all download, load, and generate correctly in bundled builds
- **Real-time download progress** — byte-level progress bars now work in production. The root cause: `huggingface_hub` silently disables tqdm progress bars based on logger level, which prevented our progress tracker from receiving byte updates. We now force-enable the internal counter regardless.
- **Fixed Python 3.12.0 `code.replace()` bug** — the macOS build was on Python 3.12.0, which has a [known CPython bug](https://github.com/pyinstaller/pyinstaller/issues/7992) that corrupts bytecode when PyInstaller rewrites code objects. This caused `NameError: name 'obj' is not defined` crashes during scipy/torch imports. Upgraded to Python 3.12.13.

### PyInstaller Fixes

- Collect all `inflect` files — `typeguard`'s `@typechecked` decorator calls `inspect.getsource()` at import time, which needs `.py` source files, not just bytecode. Fixes LuxTTS "could not get source code" error.
- Collect all `perth` files — bundles the pretrained watermark model (`hparams.yaml`, `.pth.tar`) needed by Chatterbox at runtime
- Collect all `piper_phonemize` files — bundles `espeak-ng-data/` (phoneme tables, language dicts) needed by LuxTTS for text-to-phoneme conversion
- Set `ESPEAK_DATA_PATH` in frozen builds so the espeak-ng C library finds the bundled data instead of looking at `/usr/share/espeak-ng-data/`
- Collect all `linacodec` files — fixes `inspect.getsource` error in Vocos codec
- Collect all `zipvoice` files — fixes source code lookup in LuxTTS voice cloning
- Copy metadata for `requests`, `transformers`, `huggingface-hub`, `tokenizers`, `safetensors`, `tqdm` — fixes `importlib.metadata` lookups in frozen binary
- Add hidden imports for `chatterbox`, `chatterbox_turbo`, `luxtts`, `zipvoice` backends
- Add `multiprocessing.freeze_support()` to fix resource_tracker subprocess crash in frozen binary
- `--noconsole` now only applied on Windows — macOS/Linux need stdout/stderr for Tauri sidecar log capture
- Hardened `sys.stdout`/`sys.stderr` devnull redirect to test writability, not just `None` check

### Updater

- Fixed updater artifact generation with `v1Compatible` for `tauri-action` signature files
- Updated `tauri-action` to v0.6 to fix updater JSON and `.sig` generation

### Other Fixes

- Full traceback logging on all backend model loading errors (was just `str(e)` before)

## [0.2.2] - 2026-03-15

- Fix Chatterbox model support in bundled builds
- Fix LuxTTS/ZipVoice support in bundled builds
- Auto-update CUDA binary when app version changes
- CUDA download progress bar
- Fix server process staying alive on macOS (SIGHUP handling, watchdog grace period)
- Hide console window when running CUDA binary on Windows

## [0.2.1] - 2026-03-15

Voicebox v0.1.x was a single-engine voice cloning app built around Qwen3-TTS. v0.2.0 is a ground-up rethink: four TTS engines, 23 languages, paralinguistic emotion controls, a post-processing effects pipeline, unlimited generation length, an async generation queue, and support for every major GPU vendor. Plus Docker.

### New TTS Engines

#### Multi-Engine Architecture

Voicebox now runs **four independent TTS engines** behind a thread-safe per-engine backend registry. Switch engines per-generation from a single dropdown — no restart required.

| Engine                      | Languages | Size    | Key Strengths                                 |
| --------------------------- | --------- | ------- | --------------------------------------------- |
| **Qwen3-TTS 1.7B**          | 10        | ~3.5 GB | Highest quality, delivery instructions        |
| **Qwen3-TTS 0.6B**          | 10        | ~1.2 GB | Lighter, faster variant                       |
| **LuxTTS**                  | English   | ~300 MB | CPU-friendly, 48 kHz output, 150x realtime    |
| **Chatterbox Multilingual** | 23        | ~3.2 GB | Broadest language coverage, zero-shot cloning |
| **Chatterbox Turbo**        | English   | ~1.5 GB | 350M params, low latency, paralinguistic tags |

#### Chatterbox Multilingual — 23 Languages ([#257](https://github.com/jamiepine/voicebox/pull/257))

Zero-shot voice cloning in Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, and Turkish.

#### LuxTTS — Lightweight English TTS ([#254](https://github.com/jamiepine/voicebox/pull/254))

A fast, CPU-friendly English engine. ~300 MB download, 48 kHz output, runs at 150x realtime on CPU.

#### Chatterbox Turbo — Expressive English ([#258](https://github.com/jamiepine/voicebox/pull/258))

A fast 350M-parameter English model with inline paralinguistic tags.

#### Paralinguistic Tags Autocomplete ([#265](https://github.com/jamiepine/voicebox/pull/265))

Type `/` in the text input with Chatterbox Turbo selected to open an autocomplete for **9 expressive tags**: `[laugh]` `[chuckle]` `[gasp]` `[cough]` `[sigh]` `[groan]` `[sniff]` `[shush]` `[clear throat]`

### Generation

#### Unlimited Generation Length — Auto-Chunking ([#266](https://github.com/jamiepine/voicebox/pull/266))

Long text is now automatically split at sentence boundaries, generated per-chunk, and crossfaded back together. Engine-agnostic.

- Auto-chunking limit slider — 100–5,000 chars (default 800)
- Crossfade slider — 0–200ms (default 50ms)
- Max text length raised to 50,000 characters
- Smart splitting respects abbreviations, CJK punctuation, and `[tags]`

#### Asynchronous Generation Queue ([#269](https://github.com/jamiepine/voicebox/pull/269))

Generation is now fully non-blocking. Serial execution queue prevents GPU contention. Real-time SSE status streaming.

#### Generation Versions

Every generation now supports multiple versions with provenance tracking — original, effects versions, takes, source tracking, version pinning in stories, and favorites.

### Post-Processing Effects ([#271](https://github.com/jamiepine/voicebox/pull/271))

A full audio effects system powered by Spotify's `pedalboard` library: Pitch Shift, Reverb, Delay, Chorus/Flanger, Compressor, Gain, High-Pass Filter, Low-Pass Filter. 4 built-in presets, custom presets, per-profile default effects, and live preview.

### Platform Support

- **Windows Support** ([#272](https://github.com/jamiepine/voicebox/pull/272)) — Full Windows support with CUDA GPU detection
- **Linux** ([#262](https://github.com/jamiepine/voicebox/pull/262)) — AMD ROCm, NVIDIA GBM fix, WebKitGTK mic access (build from source)
- **NVIDIA CUDA Backend Swap** ([#252](https://github.com/jamiepine/voicebox/pull/252)) — Download and swap in CUDA backend from within the app
- **Intel Arc (XPU) and DirectML** — PyTorch backend supports Intel Arc and DirectML
- **Docker + Web Deployment** ([#161](https://github.com/jamiepine/voicebox/pull/161)) — 3-stage build, non-root runtime, health checks
- **Whisper Turbo** — Added `openai/whisper-large-v3-turbo` as a transcription model option

### Model Management ([#268](https://github.com/jamiepine/voicebox/pull/268))

Per-model unload, custom models directory, model folder migration, download cancel/clear UI ([#238](https://github.com/jamiepine/voicebox/pull/238)), restructured settings UI.

### Security & Reliability

- CORS hardening ([#88](https://github.com/jamiepine/voicebox/pull/88))
- Network access toggle ([#133](https://github.com/jamiepine/voicebox/pull/133))
- Offline crash fix ([#152](https://github.com/jamiepine/voicebox/pull/152))
- Atomic audio saves ([#263](https://github.com/jamiepine/voicebox/pull/263))
- Filesystem health endpoint
- Chatterbox float64 dtype fix ([#264](https://github.com/jamiepine/voicebox/pull/264))

### Accessibility ([#243](https://github.com/jamiepine/voicebox/pull/243))

Screen reader support, keyboard navigation, state-aware `aria-label` attributes on all interactive controls.

### UI Polish

- Redesigned landing page ([#274](https://github.com/jamiepine/voicebox/pull/274))
- Voices tab overhaul with inline inspector
- Responsive layout improvements
- Duplicate profile name validation ([#175](https://github.com/jamiepine/voicebox/pull/175))

### Community Contributors

[@haosenwang1018](https://github.com/haosenwang1018), [@Balneario-de-Cofrentes](https://github.com/Balneario-de-Cofrentes), [@ageofalgo](https://github.com/ageofalgo), [@mikeswann](https://github.com/mikeswann), [@rayl15](https://github.com/rayl15), [@mpecanha](https://github.com/mpecanha), [@ways2read](https://github.com/ways2read), [@ieguiguren](https://github.com/ieguiguren), [@Vaibhavee89](https://github.com/Vaibhavee89), [@pandego](https://github.com/pandego), [@luminest-llc](https://github.com/luminest-llc)

## [0.1.13] - 2026-02-23

### Stability and reliability

- [#95](https://github.com/jamiepine/voicebox/pull/95) Fix: selecting 0.6B model still downloads and uses 1.7B
- [#93](https://github.com/jamiepine/voicebox/pull/93) fix(mlx): bundle native libs and broaden error handling for Apple Silicon
- [#79](https://github.com/jamiepine/voicebox/pull/79) fix: handle non-ASCII filenames in Content-Disposition headers
- [#78](https://github.com/jamiepine/voicebox/pull/78) fix: guard getUserMedia call against undefined mediaDevices in non-secure contexts
- [#77](https://github.com/jamiepine/voicebox/pull/77) fix: await for confirmation before deleting voices and channels
- [#128](https://github.com/jamiepine/voicebox/pull/128) fix: resolve multiple issues (#96, #119, #111, #108, #121, #125, #127)
- [#40](https://github.com/jamiepine/voicebox/pull/40) Fix: audio export path resolution

### Build and packaging

- [#122](https://github.com/jamiepine/voicebox/pull/122) fix(web): add @tailwindcss/vite plugin to web config
- [#126](https://github.com/jamiepine/voicebox/pull/126) Create requirements.txt

### UX and docs

- [#44](https://github.com/jamiepine/voicebox/pull/44) Enhances floating generate box UX
- [#57](https://github.com/jamiepine/voicebox/pull/57) chore: updates repo URL in README
- [#146](https://github.com/jamiepine/voicebox/pull/146) Add Spacebot banner to landing page
- [#1](https://github.com/jamiepine/voicebox/pull/1) Improvements

## [0.1.12] - 2026-01-31

### Model Download UX Overhaul

- Real-time download progress tracking with accurate percentage and speed info
- No more downloading notifications during generation even when its not downloading
- Better error handling and status reporting throughout the download process

### Other Improvements

- Enhanced health check endpoint with GPU type information
- Improved model caching verification
- More reliable SSE progress updates
- Actual update notifications — no need to manually check in settings anymore

## [0.1.11] - 2026-01-30

- Fixed transcriptions on MLX
- Fixed model download progress (finally)

## [0.1.10] - 2026-01-30

### Faster generation on Apple Silicon

Massive speed gains, from around 20s per generation to 2-3s. Added native MLX backend support for Apple Silicon, providing significantly faster TTS and STT generation on M-series macOS machines.

- **MLX Backend** — New backend implementation optimized for Apple Silicon using MLX framework
- **Dynamic Backend Selection** — Automatically detects platform and selects between MLX (macOS) and PyTorch (other platforms)
- Refactored TTS and STT logic into modular backend implementations
- Updated build process to include MLX-specific dependencies for macOS builds

## [0.1.9] - 2026-01-30

### Improved voice profile creation flow

- Voice create drafts: No longer lose work if you close the modal
- Fixed whisper only transcribing English or Chinese, now has support for all languages

### Improved Stories editor

- Added spacebar for play/pause
- Timeline now auto-scrolls to follow playhead during playback
- Fixed misalignment of the items with mouse when picking up
- Fixed hitbox for selecting an item
- Fixed playhead jumping forward when pressing play

### Generation box improvements

- Instruct mode no longer wipes prompt text
- Improved UI cleanliness

### Misc

- Fixed "Model downloading" toast during generation when model is already downloaded

## [0.1.8] - 2026-01-29

### Model Download Timeout Issues

Fixed critical issue where model downloads would fail with "Failed to fetch" errors on Windows. Refactored download endpoints to return immediately and continue downloads in background.

### Cross-Platform Cache Path Issues

Fixed hardcoded `~/.cache/huggingface/hub` paths that don't work on Windows. All cache paths now use `hf_constants.HF_HUB_CACHE` for proper cross-platform support.

### Windows Process Management

- Added `/shutdown` endpoint for graceful server shutdown on Windows
- Added `gpu_type` field to health check response

## [0.1.7] - 2026-01-29

- Trim and split audio clips in Story Editor
- Auto-activation of stories in Story Editor with visible playhead
- Conditional auto-play support in AudioPlayer for better user control
- Refactored audio loading across HistoryTable, SampleList, and generation forms
- Audio now only auto-plays when explicitly intended, preventing unexpected playback

## [0.1.6] - 2026-01-29

### Introducing Stories

A full voice editor for composing podcasts and generated conversations.

- **Stories Editor** — Create multi-voice narratives, podcasts, or conversations with a timeline-based editor
- Compose tracks with different voices
- Edit and arrange audio segments inline
- Build generated conversations with multiple participants
- **Improved Voice Generation UI** — Auto-resizing input, default voice selection, better layout
- **Track Editor Integration** — Inline track editing within story items

## [0.1.5] - 2026-01-28

Fixed recording length limit at 0:29 to auto stop instead of passing the limit and getting an error, which would cause users to lose their recording.

## [0.1.4] - 2026-01-28

- Audio channel management system
- Native audio playback handling in AudioPlayer component
- Refactored ConnectionForm and Checkbox components
- Improved layout consistency and responsiveness
- Added safe area constants for better responsive design

## [0.1.3] - 2026-01-27

- Improved the generate textbox
- Maybe fixed Windows autoupdate restarting entire computer

## [0.1.2] - 2026-01-27

### Audio Capture & Format Conversion

- Added audio format conversion util
- Enhanced system audio capture on macOS and Windows
- Improved audio recording hooks
- Added audio input entitlement for macOS
- Added audio capture tests

### Update System

- Enhanced auto-updater functionality and update status display

## [0.1.1] - 2026-01-27

### Platform Support

- **macOS Audio Capture** — Native audio capture support for sample creation
- **Windows Audio Capture** — WASAPI implementation with improved thread safety
- **Linux Support** — Temporarily removed builds due to runner disk space constraints

### Audio Features

- Play/pause for audio samples across all components
- Three new sample components: Recording, System capture, Upload with drag-and-drop
- Audio validation, error handling, and consistent cleanup

### Voice Profile Management

- Profile import with file size validation (100MB limit)
- Enhanced profile form with new audio sample components
- Drag-and-drop support for audio file uploads

### Server Management

- Changed default URL from `localhost:8000` to `127.0.0.1:17493`
- Server reuse logic, "keep server running" preference, orphaned process handling

### Build & Release

- Added `.bumpversion.cfg` for automated version management
- Enhanced icon generation script for multi-size Windows icons

### Bug Fixes

- Fixed date formatting for timezone-less date strings
- Fixed getLatestRelease file filtering
- Improved audio duration metadata on Windows

## [0.1.0] - 2026-01-27

The first public release of Voicebox — an open-source voice synthesis studio powered by Qwen3-TTS.

### Voice Cloning with Qwen3-TTS

- Automatic model download from HuggingFace
- Multiple model sizes (1.7B and 0.6B)
- Voice prompt caching for instant regeneration
- English and Chinese support

### Voice Profile Management

- Create profiles from audio files or record directly in the app
- Multiple samples per profile for higher quality cloning
- Import/Export profiles
- Automatic transcription via Whisper

### Speech Generation

- Simple text-to-speech with profile selection
- Seed control for reproducible generations
- Long-form support up to 5,000 characters

### Generation History

- Full history with metadata
- Search by text content
- Inline playback and download

### Flexible Deployment

- Local mode with bundled backend
- Remote mode for GPU servers on your network
- One-click server setup

### Desktop Experience

- Built with Tauri v2 (Rust) — native performance, not Electron
- Cross-platform: macOS and Windows
- No Python installation required

### Tech Stack

Tauri v2, React, TypeScript, Tailwind CSS, FastAPI, Qwen3-TTS, Whisper, SQLite

[0.5.0]: https://github.com/jamiepine/voicebox/compare/v0.4.5...v0.5.0
[0.4.5]: https://github.com/jamiepine/voicebox/compare/v0.4.4...v0.4.5
[0.4.4]: https://github.com/jamiepine/voicebox/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/jamiepine/voicebox/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/jamiepine/voicebox/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/jamiepine/voicebox/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/jamiepine/voicebox/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jamiepine/voicebox/compare/v0.2.3...v0.3.0
[0.2.3]: https://github.com/jamiepine/voicebox/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/jamiepine/voicebox/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/jamiepine/voicebox/compare/v0.1.13...v0.2.1
[0.1.13]: https://github.com/jamiepine/voicebox/compare/v0.1.12...v0.1.13
[0.1.12]: https://github.com/jamiepine/voicebox/compare/v0.1.11...v0.1.12
[0.1.11]: https://github.com/jamiepine/voicebox/compare/v0.1.10...v0.1.11
[0.1.10]: https://github.com/jamiepine/voicebox/compare/v0.1.9...v0.1.10
[0.1.9]: https://github.com/jamiepine/voicebox/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/jamiepine/voicebox/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/jamiepine/voicebox/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/jamiepine/voicebox/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/jamiepine/voicebox/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/jamiepine/voicebox/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/jamiepine/voicebox/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/jamiepine/voicebox/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/jamiepine/voicebox/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/jamiepine/voicebox/releases/tag/v0.1.0
