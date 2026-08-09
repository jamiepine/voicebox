import type {
  CaptureListResponse,
  CaptureReadinessResponse,
  CaptureResponse,
  CaptureSettings,
  EffectPresetResponse,
  GenerationResponse,
  GenerationSettings,
  HealthResponse,
  HistoryListResponse,
  HistoryResponse,
  ModelStatus,
  StoryDetailResponse,
  StoryItemDetail,
  StoryResponse,
  VoiceProfileResponse,
} from '@/lib/api/types';

// Deterministic id counter — no randomness so failures reproduce exactly.
let seq = 0;
export function nextId(prefix: string): string {
  seq += 1;
  return `${prefix}-${String(seq).padStart(4, '0')}`;
}

const CREATED_AT = '2026-01-01T00:00:00Z';

export function buildProfile(overrides: Partial<VoiceProfileResponse> = {}): VoiceProfileResponse {
  return {
    id: nextId('profile'),
    name: 'Test Voice',
    language: 'en',
    voice_type: 'cloned',
    generation_count: 0,
    sample_count: 1,
    created_at: CREATED_AT,
    updated_at: CREATED_AT,
    ...overrides,
  };
}

export function buildGeneration(overrides: Partial<GenerationResponse> = {}): GenerationResponse {
  return {
    id: nextId('gen'),
    profile_id: 'profile-0001',
    text: 'Hello from the test suite.',
    language: 'en',
    status: 'completed',
    audio_path: '/audio/fake.wav',
    duration: 1.5,
    created_at: CREATED_AT,
    ...overrides,
  };
}

export function buildHistoryItem(overrides: Partial<HistoryResponse> = {}): HistoryResponse {
  return {
    ...buildGeneration(),
    profile_name: 'Test Voice',
    ...overrides,
  };
}

export function buildHistoryList(items: HistoryResponse[]): HistoryListResponse {
  return { items, total: items.length };
}

export function buildCapture(overrides: Partial<CaptureResponse> = {}): CaptureResponse {
  return {
    id: nextId('capture'),
    audio_path: '/captures/fake.wav',
    source: 'dictation',
    language: 'en',
    duration_ms: 2400,
    transcript_raw: 'raw transcript text',
    transcript_refined: 'Refined transcript text.',
    created_at: CREATED_AT,
    ...overrides,
  };
}

export function buildCaptureList(items: CaptureResponse[]): CaptureListResponse {
  return { items, total: items.length };
}

export function buildCaptureSettings(overrides: Partial<CaptureSettings> = {}): CaptureSettings {
  return {
    stt_model: 'turbo',
    language: 'en',
    auto_refine: true,
    llm_model: '0.6B',
    smart_cleanup: true,
    self_correction: true,
    preserve_technical: true,
    allow_auto_paste: false,
    default_playback_voice_id: null,
    hotkey_enabled: false,
    keep_mic_warm: false,
    chord_push_to_talk_keys: [],
    chord_toggle_to_talk_keys: [],
    ...overrides,
  };
}

export function buildCaptureReadiness(
  overrides: Partial<CaptureReadinessResponse> = {},
): CaptureReadinessResponse {
  return {
    stt: {
      ready: true,
      model_name: 'whisper-turbo',
      display_name: 'Whisper Turbo',
      size: '1.6 GB',
    },
    llm: {
      ready: true,
      model_name: 'qwen3-0.6b',
      display_name: 'Qwen3 0.6B',
      size: '600 MB',
    },
    ...overrides,
  };
}

export function buildGenerationSettings(
  overrides: Partial<GenerationSettings> = {},
): GenerationSettings {
  return {
    max_chunk_chars: 400,
    crossfade_ms: 60,
    normalize_audio: true,
    autoplay_on_generate: false,
    ...overrides,
  };
}

export function buildModelStatus(overrides: Partial<ModelStatus> = {}): ModelStatus {
  return {
    model_name: 'qwen-tts-1.7b',
    display_name: 'Qwen TTS 1.7B',
    downloaded: true,
    downloading: false,
    loaded: false,
    size_mb: 3400,
    ...overrides,
  };
}

export function buildStory(overrides: Partial<StoryResponse> = {}): StoryResponse {
  return {
    id: nextId('story'),
    name: 'Test Story',
    created_at: CREATED_AT,
    updated_at: CREATED_AT,
    item_count: 0,
    ...overrides,
  };
}

export function buildStoryItem(overrides: Partial<StoryItemDetail> = {}): StoryItemDetail {
  return {
    id: nextId('story-item'),
    story_id: 'story-0001',
    generation_id: 'gen-0001',
    start_time_ms: 0,
    track: 0,
    trim_start_ms: 0,
    trim_end_ms: 0,
    created_at: CREATED_AT,
    profile_id: 'profile-0001',
    profile_name: 'Test Voice',
    text: 'Hello from the test suite.',
    language: 'en',
    audio_path: '/audio/fake.wav',
    duration: 1.5,
    volume: 1,
    generation_created_at: CREATED_AT,
    ...overrides,
  };
}

export function buildStoryDetail(
  overrides: Partial<StoryDetailResponse> = {},
): StoryDetailResponse {
  return {
    id: 'story-0001',
    name: 'Test Story',
    created_at: CREATED_AT,
    updated_at: CREATED_AT,
    items: [],
    ...overrides,
  };
}

export function buildEffectPreset(
  overrides: Partial<EffectPresetResponse> = {},
): EffectPresetResponse {
  return {
    id: nextId('preset'),
    name: 'Test Preset',
    effects_chain: [{ type: 'reverb', enabled: true, params: { wet: 0.3 } }],
    is_builtin: false,
    created_at: CREATED_AT,
    ...overrides,
  };
}

export function buildHealth(overrides: Partial<HealthResponse> = {}): HealthResponse {
  return {
    status: 'ok',
    model_loaded: false,
    gpu_available: false,
    backend_variant: 'cpu',
    ...overrides,
  };
}
