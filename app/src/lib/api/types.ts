// API Types matching backend Pydantic models
import type { LanguageCode } from '@/lib/constants/languages';

export interface VoiceProfileCreate {
  name: string;
  description?: string;
  language: LanguageCode;
}

export interface VoiceProfileResponse {
  id: string;
  name: string;
  description?: string;
  language: string;
  avatar_path?: string;
  created_at: string;
  updated_at: string;
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

export interface GenerationRequest {
  profile_id: string;
  text: string;
  language: LanguageCode;
  seed?: number;
  model_size?: '1.7B' | '0.6B';
  engine?: 'qwen' | 'luxtts' | 'chatterbox' | 'chatterbox_turbo';
  instruct?: string;
}

export interface GenerationResponse {
  id: string;
  profile_id: string;
  text: string;
  language: string;
  audio_path: string;
  duration: number;
  seed?: number;
  created_at: string;
}

export interface HistoryQuery {
  profile_id?: string;
  search?: string;
  limit?: number;
  offset?: number;
}

export interface HistoryResponse extends GenerationResponse {
  profile_name: string;
}

export interface HistoryListResponse {
  items: HistoryResponse[];
  total: number;
}

export interface TranscriptionRequest {
  language?: LanguageCode;
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
  backend_variant?: string; // "cpu" or "cuda"
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
  binary_path?: string;
  downloading: boolean; // Download in progress
  download_progress?: CudaDownloadProgress;
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
  generation_created_at: string;
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
