// API Types matching backend Pydantic models
import type { LanguageCode } from '@/lib/constants/languages';

export type VoiceType = 'cloned' | 'preset' | 'designed';

export interface VoiceProfileCreate {
  name: string;
  description?: string;
  language: LanguageCode;
  voice_type?: VoiceType;
  preset_engine?: string;
  preset_voice_id?: string;
  design_prompt?: string;
  default_engine?: string;
  /** Free-form character prompt used by compose and the `/generate` personality-rewrite path. */
  personality?: string;
}

export interface VoiceProfileResponse {
  id: string;
  name: string;
  description?: string;
  language: string;
  avatar_path?: string;
  effects_chain?: EffectConfig[];
  voice_type: VoiceType;
  preset_engine?: string;
  preset_voice_id?: string;
  design_prompt?: string;
  default_engine?: string;
  personality?: string | null;
  generation_count: number;
  sample_count: number;
  created_at: string;
  updated_at: string;
}

/** Response returned by /profiles/{id}/compose. */
export interface PersonalityTextResponse {
  text: string;
  model_size: string;
}

export interface PresetVoice {
  voice_id: string;
  name: string;
  gender: 'male' | 'female';
  language: string;
}

export interface ProfileSampleCreate {
  reference_text: string;
}

export interface ProfileSampleResponse {
  id: string;
  profile_id: string;
  audio_path: string;
  reference_text: string;
}

export interface EffectConfig {
  type: string;
  enabled: boolean;
  params: Record<string, number>;
}

export interface GenerationRequest {
  profile_id: string;
  text: string;
  language: LanguageCode;
  seed?: number;
  model_size?: '1.7B' | '0.6B' | '1B' | '3B';
  engine?:
    | 'qwen'
    | 'qwen_custom_voice'
    | 'luxtts'
    | 'chatterbox'
    | 'chatterbox_turbo'
    | 'tada'
    | 'kokoro';
  instruct?: string;
  /** When true and the profile has a personality prompt, input text is rewritten in-character before TTS. */
  personality?: boolean;
  max_chunk_chars?: number;
  crossfade_ms?: number;
  normalize?: boolean;
  effects_chain?: EffectConfig[];
}

export interface GenerationVersionResponse {
  id: string;
  generation_id: string;
  label: string;
  audio_path: string;
  effects_chain?: EffectConfig[];
  source_version_id?: string;
  is_default: boolean;
  created_at: string;
}

export interface GenerationResponse {
  id: string;
  profile_id: string;
  text: string;
  language: string;
  audio_path?: string;
  duration?: number;
  seed?: number;
  instruct?: string;
  engine?: string;
  model_size?: string;
  status: 'loading_model' | 'generating' | 'completed' | 'failed';
  error?: string;
  is_favorited?: boolean;
  created_at: string;
  versions?: GenerationVersionResponse[];
  active_version_id?: string;
}

export interface HistoryQuery {
  profile_id?: string;
  search?: string;
  limit?: number;
  offset?: number;
}

export interface HistoryResponse extends GenerationResponse {
  profile_name: string;
  versions?: GenerationVersionResponse[];
  active_version_id?: string;
}

export interface HistoryListResponse {
  items: HistoryResponse[];
  total: number;
}

export type WhisperModelSize = 'base' | 'small' | 'medium' | 'large' | 'turbo';

export type Qwen3ModelSize = '0.6B' | '1.7B' | '4B';

export type CaptureSource = 'dictation' | 'recording' | 'file';

/**
 * Snapshot of the accessibility-focused UI element at chord-start. Emitted
 * from Rust as part of the ``dictate:start`` payload so the frontend can
 * pass it back to ``paste_final_text`` once the final text is ready.
 */
export interface FocusSnapshot {
  pid: number;
  bundle_id: string | null;
  role: string | null;
}

export interface RefinementFlags {
  smart_cleanup: boolean;
  self_correction: boolean;
  preserve_technical: boolean;
}

export interface CaptureResponse {
  id: string;
  audio_path: string;
  source: CaptureSource;
  language?: string | null;
  duration_ms?: number | null;
  transcript_raw: string;
  transcript_refined?: string | null;
  stt_model?: string | null;
  llm_model?: string | null;
  refinement_flags?: RefinementFlags | null;
  created_at: string;
}

export interface CaptureListResponse {
  items: CaptureResponse[];
  total: number;
}

/**
 * Response of ``POST /captures``. Adds ``auto_refine`` and ``allow_auto_paste``
 * — the server's current settings captured at request time — so the client
 * can decide whether to chain a refine call and whether to fire the
 * synthetic-paste pipeline without relying on its own (possibly stale) copy
 * of capture_settings.
 */
export interface CaptureCreateResponse extends CaptureResponse {
  auto_refine: boolean;
  allow_auto_paste: boolean;
}

export interface CaptureRefineRequest {
  flags?: RefinementFlags;
  model_size?: Qwen3ModelSize;
}

export interface CaptureRetranscribeRequest {
  model?: WhisperModelSize;
  language?: LanguageCode;
}

export interface CaptureSettings {
  stt_model: WhisperModelSize;
  language: string;
  auto_refine: boolean;
  llm_model: Qwen3ModelSize;
  smart_cleanup: boolean;
  self_correction: boolean;
  preserve_technical: boolean;
  allow_auto_paste: boolean;
  default_playback_voice_id: string | null;
  /** Whether the global keyboard hotkey is armed. Off by default — turning
   *  this on triggers the macOS Input Monitoring TCC prompt. */
  hotkey_enabled: boolean;
  /** keytap key names. Defaults are platform-specific right-hand modifiers. */
  chord_push_to_talk_keys: string[];
  /** keytap key names. Toggle adds Space to the platform-specific PTT chord. */
  chord_toggle_to_talk_keys: string[];
}

export type CaptureSettingsUpdate = Partial<CaptureSettings>;

/**
 * One row in the dictation readiness checklist. ``model_name`` is the
 * canonical id understood by ``POST /models/download`` so the UI can wire a
 * one-click "Download" button without a second lookup.
 */
export interface ModelReadiness {
  ready: boolean;
  model_name: string;
  display_name: string;
  size: string;
  size_mb?: number | null;
}

/** Backend half of the dictation readiness check. The frontend combines this
 *  with TCC permission state into the full checklist used by useDictationReadiness. */
export interface CaptureReadinessResponse {
  stt: ModelReadiness;
  llm: ModelReadiness;
}

export interface GenerationSettings {
  max_chunk_chars: number;
  crossfade_ms: number;
  normalize_audio: boolean;
  autoplay_on_generate: boolean;
}

export type GenerationSettingsUpdate = Partial<GenerationSettings>;

export interface TranscriptionRequest {
  language?: LanguageCode;
  model?: WhisperModelSize;
}

export interface TranscriptionResponse {
  text: string;
  duration: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_downloaded?: boolean;
  model_size?: string;
  gpu_available: boolean;
  gpu_type?: string;
  vram_used_mb?: number;
  backend_type?: string;
  backend_variant?: string; // "cpu", "cuda", or "rocm"
  supports_rocm?: boolean; // AMD GPU on Windows — the ROCm backend is applicable
}

export interface CudaDownloadProgress {
  model_name: string;
  current: number;
  total: number;
  progress: number;
  filename?: string;
  status: 'downloading' | 'extracting' | 'complete' | 'error';
  timestamp: string;
  error?: string;
}

export interface CudaStatus {
  available: boolean; // CUDA binary exists on disk
  active: boolean; // Currently running the CUDA binary
  binary_path: string | null;
  cuda_libs_version: string | null;
  download_supported: boolean; // Platform has a matching release asset
  unsupported_reason: string | null;
  downloading: boolean; // Download in progress
  download_progress?: CudaDownloadProgress;
}

export interface RocmDownloadProgress {
  model_name: string;
  current: number;
  total: number;
  progress: number;
  filename?: string;
  status: 'downloading' | 'extracting' | 'complete' | 'error';
  timestamp: string;
  error?: string;
}

export interface RocmStatus {
  available: boolean; // ROCm binary exists on disk
  active: boolean; // Currently running the ROCm binary
  binary_path?: string;
  rocm_libs_version?: string;
  downloading: boolean; // Download in progress
  download_progress?: RocmDownloadProgress;
}

export interface ModelProgress {
  model_name: string;
  current: number;
  total: number;
  progress: number;
  filename?: string;
  status: 'downloading' | 'extracting' | 'complete' | 'error';
  timestamp: string;
  error?: string;
}

export interface ModelStatus {
  model_name: string;
  display_name: string;
  hf_repo_id?: string; // HuggingFace repository ID
  downloaded: boolean;
  downloading: boolean; // True if download is in progress
  size_mb?: number;
  loaded: boolean;
}

export interface HuggingFaceModelInfo {
  id: string;
  author: string;
  lastModified: string;
  pipeline_tag?: string;
  library_name?: string;
  downloads: number;
  likes: number;
  tags: string[];
  cardData?: {
    license?: string;
    language?: string[];
    pipeline_tag?: string;
  };
}

export interface ModelStatusListResponse {
  models: ModelStatus[];
}

export interface ModelDownloadRequest {
  model_name: string;
}

export interface ActiveDownloadTask {
  model_name: string;
  status: string;
  started_at: string;
  error?: string;
  progress?: number; // 0-100 percentage
  current?: number; // bytes downloaded
  total?: number; // total bytes
  filename?: string; // current file being downloaded
}

export interface ActiveGenerationTask {
  task_id: string;
  profile_id: string;
  text_preview: string;
  started_at: string;
}

export interface ActiveTasksResponse {
  downloads: ActiveDownloadTask[];
  generations: ActiveGenerationTask[];
}

export interface StoryCreate {
  name: string;
  description?: string;
}

export interface StoryResponse {
  id: string;
  name: string;
  description?: string;
  created_at: string;
  updated_at: string;
  item_count: number;
}

export interface StoryItemDetail {
  id: string;
  story_id: string;
  generation_id: string;
  version_id?: string;
  start_time_ms: number;
  track: number;
  trim_start_ms: number;
  trim_end_ms: number;
  created_at: string;
  profile_id: string;
  profile_name: string;
  text: string;
  language: string;
  audio_path: string;
  duration: number;
  seed?: number;
  instruct?: string;
  engine?: string;
  volume: number;
  generation_created_at: string;
  versions?: GenerationVersionResponse[];
  active_version_id?: string;
}

export interface StoryItemVolumeUpdate {
  volume: number;
}

export interface StoryItemVersionUpdate {
  version_id: string | null;
}

export interface StoryDetailResponse {
  id: string;
  name: string;
  description?: string;
  created_at: string;
  updated_at: string;
  items: StoryItemDetail[];
}

export interface StoryItemCreate {
  generation_id: string;
  start_time_ms?: number;
  track?: number;
}

export interface StoryItemUpdateTime {
  generation_id: string;
  start_time_ms: number;
}

export interface StoryItemBatchUpdate {
  updates: StoryItemUpdateTime[];
}

export interface StoryItemReorder {
  generation_ids: string[];
}

export interface StoryItemMove {
  start_time_ms: number;
  track: number;
}

export interface StoryItemTrim {
  trim_start_ms: number;
  trim_end_ms: number;
}

export interface StoryItemSplit {
  split_time_ms: number;
}

// Effects

export interface EffectPresetResponse {
  id: string;
  name: string;
  description?: string;
  effects_chain: EffectConfig[];
  is_builtin: boolean;
  created_at: string;
}

export interface EffectPresetCreate {
  name: string;
  description?: string;
  effects_chain: EffectConfig[];
}

export interface EffectPresetUpdate {
  name?: string;
  description?: string;
  effects_chain?: EffectConfig[];
}

export interface AvailableEffectParam {
  default: number;
  min: number;
  max: number;
  step: number;
  description: string;
}

export interface AvailableEffect {
  type: string;
  label: string;
  description: string;
  params: Record<string, AvailableEffectParam>;
}

export interface AvailableEffectsResponse {
  effects: AvailableEffect[];
}

export interface ApplyEffectsRequest {
  effects_chain: EffectConfig[];
  source_version_id?: string;
  label?: string;
  set_as_default?: boolean;
}

/* ─── MCP ─────────────────────────────────────────────────────────────── */

export interface MCPClientBinding {
  client_id: string;
  label: string | null;
  profile_id: string | null;
  default_engine: string | null;
  default_personality: boolean;
  last_seen_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface MCPClientBindingUpsert {
  client_id: string;
  label?: string | null;
  profile_id?: string | null;
  default_engine?: string | null;
  default_personality?: boolean;
}

export interface MCPClientBindingListResponse {
  items: MCPClientBinding[];
}

/* ─── Cloud (backup & sync) ───────────────────────────────────────────── */

export interface CloudLoginStartResponse {
  authorize_url: string;
}

export interface CloudStatus {
  connected: boolean;
  device_name: string | null;
  account_user_id: string | null;
  key_prefix: string | null;
  connected_at: string | null;
  dashboard_url: string;
}

/* ─── Voice AI agent ──────────────────────────────────────────────────── */

export type VoiceAgentMode = 'outbound_sales' | 'customer_service' | 'support';
export type VoiceAgentStatus = 'draft' | 'active' | 'paused' | 'completed';
export type VoiceAgentProvider = 'local' | 'twilio';

export type CallOutcome =
  | 'interested'
  | 'not_interested'
  | 'callback'
  | 'opt_out'
  | 'resolved'
  | 'unresolved'
  | 'ticket_created'
  | 'handoff'
  | 'no_answer'
  | 'voicemail'
  | 'max_turns'
  | 'error';

export interface VoiceAgentCreate {
  name: string;
  mode: VoiceAgentMode;
  profile: string;
  engine?: string | null;
  language: string;
  llm_model_size?: string | null;
  agent_name: string;
  company_name: string;
  brief: string;
  goal: string;
  objection_notes?: string | null;
  persona?: string | null;
  opening_line?: string | null;
  disclosure: string;
  escalation_promise?: string | null;
  timezone: string;
  calling_window_start: number;
  calling_window_end: number;
  calling_days: number[];
  max_attempts: number;
  daily_call_cap: number;
  retry_delay_hours: number;
  callback_delay_hours: number;
  require_consent: boolean;
  max_turns: number;
  handoff_after_negative_turns: number;
  provider: VoiceAgentProvider;
  from_number?: string | null;
}

export type VoiceAgentUpdate = Partial<VoiceAgentCreate>;

export interface VoiceAgent extends Omit<VoiceAgentCreate, 'profile'> {
  id: string;
  status: VoiceAgentStatus;
  profile_id: string;
  running: boolean;
  created_at: string;
  updated_at: string;
}

export interface VoiceAgentStats {
  agent_id: string;
  mode: VoiceAgentMode;
  status: VoiceAgentStatus;
  running: boolean;
  contacts_total: number;
  contacts_by_status: Record<string, number>;
  calls_total: number;
  calls_today: number;
  calls_by_outcome: Record<string, number>;
  avg_turns: number;
  resolution_rate: number;
  open_tickets: number;
  next_dialable: number;
}

export interface ContactCreate {
  name: string;
  phone: string;
  company?: string | null;
  notes?: string | null;
  timezone?: string | null;
  consent?: boolean;
}

export interface ContactUpdate extends Partial<ContactCreate> {
  memory?: string | null;
  status?: string;
  next_attempt_at?: string | null;
}

export interface Contact {
  id: string;
  agent_id: string;
  name: string;
  phone: string;
  company: string | null;
  notes: string | null;
  memory: string | null;
  timezone: string | null;
  consent: boolean;
  status: string;
  attempts: number;
  last_attempt_at: string | null;
  next_attempt_at: string | null;
  last_outcome: string | null;
  created_at: string;
}

export interface ContactListResponse {
  contacts: Contact[];
  total: number;
}

export interface ContactImportResult {
  imported: number;
  skipped: number;
  skipped_reasons: Record<string, number>;
}

export interface KnowledgeArticleCreate {
  title: string;
  content: string;
  tags?: string[] | null;
}

export interface KnowledgeArticle extends KnowledgeArticleCreate {
  id: string;
  agent_id: string;
  tags: string[];
  created_at: string;
  updated_at: string;
}

export interface CallTurn {
  id: string;
  call_id: string;
  role: 'agent' | 'customer';
  text: string;
  sentiment: number | null;
  generation_id: string | null;
  capture_id: string | null;
  created_at: string;
}

export interface VoiceCall {
  id: string;
  agent_id: string;
  contact_id: string;
  direction: 'outbound' | 'inbound';
  status: 'in_progress' | 'completed' | 'failed';
  stage: string;
  outcome: CallOutcome | null;
  summary: string | null;
  provider: VoiceAgentProvider;
  provider_call_id: string | null;
  turn_count: number;
  started_at: string;
  ended_at: string | null;
  turns: CallTurn[];
}

export interface CallListResponse {
  calls: VoiceCall[];
  total: number;
}

export interface AgentTurnResponse {
  call_id: string;
  text: string;
  generation_id: string | null;
  poll_url: string | null;
  ended: boolean;
  outcome: CallOutcome | null;
  ticket_id: string | null;
  customer_text: string | null;
  sentiment: number | null;
}

export interface Ticket {
  id: string;
  agent_id: string;
  contact_id: string;
  call_id: string | null;
  kind: 'support' | 'handoff' | 'callback' | 'sales_lead';
  priority: 'low' | 'normal' | 'high' | 'urgent';
  status: 'open' | 'in_progress' | 'resolved' | 'closed';
  subject: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface TicketListResponse {
  tickets: Ticket[];
  total: number;
}

export interface DoNotCallEntry {
  phone: string;
  reason: string | null;
  source: string;
  created_at: string;
}
