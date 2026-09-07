import type { LanguageCode } from '@/lib/constants/languages';
import { useServerStore } from '@/stores/serverStore';
import type {
  ActiveTasksResponse,
  AgentMessage,
  AgentTool,
  AgentToolCreate,
  AgentTurnResponse,
  AnalyticsResponse,
  ApplyEffectsRequest,
  Appointment,
  AvailableEffectsResponse,
  CallListResponse,
  CaptureCreateResponse,
  CaptureListResponse,
  CaptureReadinessResponse,
  CaptureRefineRequest,
  CaptureResponse,
  CaptureRetranscribeRequest,
  CaptureSettings,
  CaptureSettingsUpdate,
  CaptureSource,
  CloudLoginStartResponse,
  CloudStatus,
  Contact,
  ContactCreate,
  ContactImportResult,
  ContactListResponse,
  ContactUpdate,
  CudaStatus,
  DoNotCallEntry,
  DoNotCallImportResult,
  EffectConfig,
  EffectPresetCreate,
  EffectPresetResponse,
  GenerationRequest,
  GenerationResponse,
  GenerationSettings,
  GenerationSettingsUpdate,
  GenerationVersionResponse,
  HealthResponse,
  HistoryListResponse,
  HistoryQuery,
  HistoryResponse,
  KnowledgeArticle,
  KnowledgeArticleCreate,
  KnowledgeSearchResult,
  MCPClientBinding,
  MCPClientBindingListResponse,
  MCPClientBindingUpsert,
  ModelDownloadRequest,
  ModelStatusListResponse,
  PersonalityTextResponse,
  PresetVoice,
  ProfileSampleResponse,
  RocmStatus,
  SimulateRequest,
  StoryCreate,
  StoryDetailResponse,
  StoryItemBatchUpdate,
  StoryItemCreate,
  StoryItemDetail,
  StoryItemMove,
  StoryItemReorder,
  StoryItemSplit,
  StoryItemTrim,
  StoryItemVersionUpdate,
  StoryItemVolumeUpdate,
  StoryResponse,
  Ticket,
  TicketListResponse,
  TranscriptionResponse,
  VoiceAgent,
  VoiceAgentCreate,
  VoiceAgentStats,
  VoiceAgentUpdate,
  VoiceAgentVersion,
  VoiceCall,
  VoiceProfileCreate,
  VoiceProfileResponse,
  WebhookDelivery,
  WhisperModelSize,
} from './types';

function formatErrorDetail(detail: unknown, fallback: string): string {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((e: Record<string, unknown>) => e.msg || e.message || JSON.stringify(e))
      .join('; ');
  }
  if (detail && typeof detail === 'object') {
    const obj = detail as Record<string, unknown>;
    if (typeof obj.message === 'string') return obj.message;
    return JSON.stringify(detail);
  }
  return fallback;
}

class ApiClient {
  private getBaseUrl(): string {
    const serverUrl = useServerStore.getState().serverUrl;
    return serverUrl;
  }

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.getBaseUrl()}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: response.statusText,
      }));
      throw new Error(formatErrorDetail(error.detail, `HTTP error! status: ${response.status}`));
    }

    return response.json();
  }

  // Health
  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health');
  }

  // Profiles
  async createProfile(data: VoiceProfileCreate): Promise<VoiceProfileResponse> {
    return this.request<VoiceProfileResponse>('/profiles', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async listProfiles(): Promise<VoiceProfileResponse[]> {
    return this.request<VoiceProfileResponse[]>('/profiles');
  }

  async getProfile(profileId: string): Promise<VoiceProfileResponse> {
    return this.request<VoiceProfileResponse>(`/profiles/${profileId}`);
  }

  async listPresetVoices(engine: string): Promise<{ engine: string; voices: PresetVoice[] }> {
    return this.request<{ engine: string; voices: PresetVoice[] }>(`/profiles/presets/${engine}`);
  }

  async updateProfile(profileId: string, data: VoiceProfileCreate): Promise<VoiceProfileResponse> {
    return this.request<VoiceProfileResponse>(`/profiles/${profileId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteProfile(profileId: string): Promise<void> {
    await this.request<void>(`/profiles/${profileId}`, {
      method: 'DELETE',
    });
  }

  // ── Personality-driven text generation ─────────────────────────────
  // Compose produces a fresh in-character utterance the UI drops into
  // the generate textarea. Rewrite now happens server-side inside
  // `/generate` when `personality: true` is passed in the request body.

  async composeWithPersonality(profileId: string): Promise<PersonalityTextResponse> {
    return this.request<PersonalityTextResponse>(`/profiles/${profileId}/compose`, {
      method: 'POST',
    });
  }

  async addProfileSample(
    profileId: string,
    file: File,
    referenceText: string,
  ): Promise<ProfileSampleResponse> {
    const url = `${this.getBaseUrl()}/profiles/${profileId}/samples`;
    const formData = new FormData();
    formData.append('file', file);
    formData.append('reference_text', referenceText);

    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: response.statusText,
      }));
      throw new Error(formatErrorDetail(error.detail, `HTTP error! status: ${response.status}`));
    }

    return response.json();
  }

  async listProfileSamples(profileId: string): Promise<ProfileSampleResponse[]> {
    return this.request<ProfileSampleResponse[]>(`/profiles/${profileId}/samples`);
  }

  async deleteProfileSample(sampleId: string): Promise<void> {
    await this.request<void>(`/profiles/samples/${sampleId}`, {
      method: 'DELETE',
    });
  }

  async updateProfileSample(
    sampleId: string,
    referenceText: string,
  ): Promise<ProfileSampleResponse> {
    return this.request<ProfileSampleResponse>(`/profiles/samples/${sampleId}`, {
      method: 'PUT',
      body: JSON.stringify({ reference_text: referenceText }),
    });
  }

  async exportProfile(profileId: string): Promise<Blob> {
    const url = `${this.getBaseUrl()}/profiles/${profileId}/export`;
    const response = await fetch(url);

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: response.statusText,
      }));
      throw new Error(formatErrorDetail(error.detail, `HTTP error! status: ${response.status}`));
    }

    return response.blob();
  }

  async importProfile(file: File): Promise<VoiceProfileResponse> {
    const url = `${this.getBaseUrl()}/profiles/import`;
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: response.statusText,
      }));
      throw new Error(formatErrorDetail(error.detail, `HTTP error! status: ${response.status}`));
    }

    return response.json();
  }

  async uploadAvatar(profileId: string, file: File): Promise<VoiceProfileResponse> {
    const url = `${this.getBaseUrl()}/profiles/${profileId}/avatar`;
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: response.statusText,
      }));
      throw new Error(formatErrorDetail(error.detail, `HTTP error! status: ${response.status}`));
    }

    return response.json();
  }

  async deleteAvatar(profileId: string): Promise<void> {
    await this.request<void>(`/profiles/${profileId}/avatar`, {
      method: 'DELETE',
    });
  }

  // Generation
  async generateSpeech(data: GenerationRequest): Promise<GenerationResponse> {
    return this.request<GenerationResponse>('/generate', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async retryGeneration(generationId: string): Promise<GenerationResponse> {
    return this.request<GenerationResponse>(`/generate/${generationId}/retry`, {
      method: 'POST',
    });
  }

  async cancelGeneration(generationId: string): Promise<{ message: string }> {
    return this.request<{ message: string }>(`/generate/${generationId}/cancel`, {
      method: 'POST',
    });
  }

  async regenerateGeneration(generationId: string): Promise<GenerationResponse> {
    return this.request<GenerationResponse>(`/generate/${generationId}/regenerate`, {
      method: 'POST',
    });
  }

  async importAudio(file: File): Promise<GenerationResponse> {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${this.getBaseUrl()}/generate/import`, {
      method: 'POST',
      body: form,
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => res.statusText);
      throw new Error(detail || `HTTP ${res.status}`);
    }
    return res.json();
  }

  async toggleFavorite(generationId: string): Promise<{ is_favorited: boolean }> {
    return this.request<{ is_favorited: boolean }>(`/history/${generationId}/favorite`, {
      method: 'POST',
    });
  }

  // History
  async listHistory(query?: HistoryQuery): Promise<HistoryListResponse> {
    const params = new URLSearchParams();
    if (query?.profile_id) params.append('profile_id', query.profile_id);
    if (query?.search) params.append('search', query.search);
    if (query?.limit) params.append('limit', query.limit.toString());
    if (query?.offset) params.append('offset', query.offset.toString());

    const queryString = params.toString();
    const endpoint = queryString ? `/history?${queryString}` : '/history';

    return this.request<HistoryListResponse>(endpoint);
  }

  async getGeneration(generationId: string): Promise<HistoryResponse> {
    return this.request<HistoryResponse>(`/history/${generationId}`);
  }

  async deleteGeneration(generationId: string): Promise<void> {
    await this.request<void>(`/history/${generationId}`, {
      method: 'DELETE',
    });
  }

  async clearFailedGenerations(): Promise<{ deleted: number }> {
    return this.request<{ deleted: number }>(`/history/failed`, {
      method: 'DELETE',
    });
  }

  async exportGeneration(generationId: string): Promise<Blob> {
    const url = `${this.getBaseUrl()}/history/${generationId}/export`;
    const response = await fetch(url);

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: response.statusText,
      }));
      throw new Error(formatErrorDetail(error.detail, `HTTP error! status: ${response.status}`));
    }

    return response.blob();
  }

  async exportGenerationAudio(generationId: string): Promise<Blob> {
    const url = `${this.getBaseUrl()}/history/${generationId}/export-audio`;
    const response = await fetch(url);

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: response.statusText,
      }));
      throw new Error(formatErrorDetail(error.detail, `HTTP error! status: ${response.status}`));
    }

    return response.blob();
  }

  async importGeneration(file: File): Promise<{
    id: string;
    profile_id: string;
    profile_name: string;
    text: string;
    message: string;
  }> {
    const url = `${this.getBaseUrl()}/history/import`;
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: response.statusText,
      }));
      throw new Error(formatErrorDetail(error.detail, `HTTP error! status: ${response.status}`));
    }

    return response.json();
  }

  // Generation status SSE
  getGenerationStatusUrl(generationId: string): string {
    return `${this.getBaseUrl()}/generate/${generationId}/status`;
  }

  // Audio
  getAudioUrl(audioId: string): string {
    return `${this.getBaseUrl()}/audio/${audioId}`;
  }

  getSampleUrl(sampleId: string): string {
    return `${this.getBaseUrl()}/samples/${sampleId}`;
  }

  // Transcription
  async transcribeAudio(
    file: File,
    language?: LanguageCode,
    model?: WhisperModelSize,
  ): Promise<TranscriptionResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (language) {
      formData.append('language', language);
    }
    if (model) {
      formData.append('model', model);
    }

    const url = `${this.getBaseUrl()}/transcribe`;
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: response.statusText,
      }));
      throw new Error(formatErrorDetail(error.detail, `HTTP error! status: ${response.status}`));
    }

    return response.json();
  }

  // Captures
  async listCaptures(limit = 50, offset = 0): Promise<CaptureListResponse> {
    return this.request<CaptureListResponse>(`/captures?limit=${limit}&offset=${offset}`);
  }

  async getCapture(captureId: string): Promise<CaptureResponse> {
    return this.request<CaptureResponse>(`/captures/${captureId}`);
  }

  async createCapture(
    file: File,
    options?: {
      source?: CaptureSource;
      language?: LanguageCode;
      sttModel?: WhisperModelSize;
    },
  ): Promise<CaptureCreateResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('source', options?.source ?? 'file');
    if (options?.language) formData.append('language', options.language);
    if (options?.sttModel) formData.append('stt_model', options.sttModel);

    const url = `${this.getBaseUrl()}/captures`;
    const response = await fetch(url, { method: 'POST', body: formData });
    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: response.statusText,
      }));
      throw new Error(formatErrorDetail(error.detail, `HTTP error! status: ${response.status}`));
    }
    return response.json();
  }

  async deleteCapture(captureId: string): Promise<{ message: string }> {
    return this.request<{ message: string }>(`/captures/${captureId}`, {
      method: 'DELETE',
    });
  }

  async refineCapture(captureId: string, body: CaptureRefineRequest): Promise<CaptureResponse> {
    return this.request<CaptureResponse>(`/captures/${captureId}/refine`, {
      method: 'POST',
      body: JSON.stringify(body),
    });
  }

  async retranscribeCapture(
    captureId: string,
    body: CaptureRetranscribeRequest,
  ): Promise<CaptureResponse> {
    return this.request<CaptureResponse>(`/captures/${captureId}/retranscribe`, {
      method: 'POST',
      body: JSON.stringify(body),
    });
  }

  getCaptureAudioUrl(captureId: string): string {
    return `${this.getBaseUrl()}/captures/${captureId}/audio`;
  }

  // Settings
  async getCaptureSettings(): Promise<CaptureSettings> {
    return this.request<CaptureSettings>('/settings/captures');
  }

  async getCaptureReadiness(): Promise<CaptureReadinessResponse> {
    return this.request<CaptureReadinessResponse>('/capture/readiness');
  }

  async updateCaptureSettings(patch: CaptureSettingsUpdate): Promise<CaptureSettings> {
    return this.request<CaptureSettings>('/settings/captures', {
      method: 'PUT',
      body: JSON.stringify(patch),
    });
  }

  async getGenerationSettings(): Promise<GenerationSettings> {
    return this.request<GenerationSettings>('/settings/generation');
  }

  async updateGenerationSettings(patch: GenerationSettingsUpdate): Promise<GenerationSettings> {
    return this.request<GenerationSettings>('/settings/generation', {
      method: 'PUT',
      body: JSON.stringify(patch),
    });
  }

  // MCP bindings — per-MCP-client voice/engine/personality mapping.
  async listMCPBindings(): Promise<MCPClientBindingListResponse> {
    return this.request<MCPClientBindingListResponse>('/mcp/bindings');
  }

  async upsertMCPBinding(data: MCPClientBindingUpsert): Promise<MCPClientBinding> {
    return this.request<MCPClientBinding>('/mcp/bindings', {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteMCPBinding(clientId: string): Promise<{ deleted: string }> {
    return this.request<{ deleted: string }>(`/mcp/bindings/${encodeURIComponent(clientId)}`, {
      method: 'DELETE',
    });
  }

  // Model Management
  async getModelStatus(): Promise<ModelStatusListResponse> {
    return this.request<ModelStatusListResponse>('/models/status');
  }

  async getModelsCacheDir(): Promise<{ path: string }> {
    return this.request<{ path: string }>('/models/cache-dir');
  }

  async migrateModels(
    destination: string,
  ): Promise<{ source: string; destination: string; moved: number; errors: string[] }> {
    return this.request('/models/migrate', {
      method: 'POST',
      body: JSON.stringify({ destination }),
    });
  }

  getMigrationProgressUrl(): string {
    return `${this.getBaseUrl()}/models/migrate/progress`;
  }

  async triggerModelDownload(modelName: string): Promise<{ message: string }> {
    console.log(
      '[API] triggerModelDownload called for:',
      modelName,
      'at',
      new Date().toISOString(),
    );
    const result = await this.request<{ message: string }>('/models/download', {
      method: 'POST',
      body: JSON.stringify({ model_name: modelName } as ModelDownloadRequest),
    });
    console.log('[API] triggerModelDownload response:', result);
    return result;
  }

  async deleteModel(modelName: string): Promise<{ message: string }> {
    return this.request<{ message: string }>(`/models/${modelName}`, {
      method: 'DELETE',
    });
  }

  async unloadModel(modelName: string): Promise<{ message: string }> {
    return this.request<{ message: string }>(`/models/${modelName}/unload`, {
      method: 'POST',
    });
  }

  async cancelDownload(modelName: string): Promise<{ message: string }> {
    return this.request<{ message: string }>('/models/download/cancel', {
      method: 'POST',
      body: JSON.stringify({ model_name: modelName } as ModelDownloadRequest),
    });
  }

  // Task Management
  async getActiveTasks(): Promise<ActiveTasksResponse> {
    return this.request<ActiveTasksResponse>('/tasks/active');
  }

  async clearAllTasks(): Promise<{ message: string }> {
    return this.request<{ message: string }>('/tasks/clear', { method: 'POST' });
  }

  // Audio Channels
  async listChannels(): Promise<
    Array<{
      id: string;
      name: string;
      is_default: boolean;
      device_ids: string[];
      created_at: string;
    }>
  > {
    return this.request('/channels');
  }

  async createChannel(data: { name: string; device_ids: string[] }): Promise<{
    id: string;
    name: string;
    is_default: boolean;
    device_ids: string[];
    created_at: string;
  }> {
    return this.request('/channels', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateChannel(
    channelId: string,
    data: {
      name?: string;
      device_ids?: string[];
    },
  ): Promise<{
    id: string;
    name: string;
    is_default: boolean;
    device_ids: string[];
    created_at: string;
  }> {
    return this.request(`/channels/${channelId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteChannel(channelId: string): Promise<{ message: string }> {
    return this.request(`/channels/${channelId}`, {
      method: 'DELETE',
    });
  }

  async getChannelVoices(channelId: string): Promise<{ profile_ids: string[] }> {
    return this.request(`/channels/${channelId}/voices`);
  }

  async setChannelVoices(channelId: string, profileIds: string[]): Promise<{ message: string }> {
    return this.request(`/channels/${channelId}/voices`, {
      method: 'PUT',
      body: JSON.stringify({ profile_ids: profileIds }),
    });
  }

  async getProfileChannels(profileId: string): Promise<{ channel_ids: string[] }> {
    return this.request(`/profiles/${profileId}/channels`);
  }

  async setProfileChannels(profileId: string, channelIds: string[]): Promise<{ message: string }> {
    return this.request(`/profiles/${profileId}/channels`, {
      method: 'PUT',
      body: JSON.stringify({ channel_ids: channelIds }),
    });
  }

  // CUDA Backend Management
  async getCudaStatus(): Promise<CudaStatus> {
    return this.request<CudaStatus>('/backend/cuda-status');
  }

  async downloadCudaBackend(): Promise<{ message: string; progress_key: string }> {
    return this.request<{ message: string; progress_key: string }>('/backend/download-cuda', {
      method: 'POST',
    });
  }

  async deleteCudaBackend(): Promise<{ message: string }> {
    return this.request<{ message: string }>('/backend/cuda', {
      method: 'DELETE',
    });
  }

  // ROCm Backend Management
  async getRocmStatus(): Promise<RocmStatus> {
    return this.request<RocmStatus>('/backend/rocm-status');
  }

  async downloadRocmBackend(): Promise<{ message: string; progress_key: string }> {
    return this.request<{ message: string; progress_key: string }>('/backend/download-rocm', {
      method: 'POST',
    });
  }

  async deleteRocmBackend(): Promise<{ message: string }> {
    return this.request<{ message: string }>('/backend/rocm', {
      method: 'DELETE',
    });
  }

  // Stories
  async listStories(): Promise<StoryResponse[]> {
    return this.request<StoryResponse[]>('/stories');
  }

  async createStory(data: StoryCreate): Promise<StoryResponse> {
    return this.request<StoryResponse>('/stories', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getStory(storyId: string): Promise<StoryDetailResponse> {
    return this.request<StoryDetailResponse>(`/stories/${storyId}`);
  }

  async updateStory(storyId: string, data: StoryCreate): Promise<StoryResponse> {
    return this.request<StoryResponse>(`/stories/${storyId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteStory(storyId: string): Promise<void> {
    await this.request<void>(`/stories/${storyId}`, {
      method: 'DELETE',
    });
  }

  async addStoryItem(storyId: string, data: StoryItemCreate): Promise<StoryItemDetail> {
    return this.request<StoryItemDetail>(`/stories/${storyId}/items`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async removeStoryItem(storyId: string, itemId: string): Promise<void> {
    await this.request<void>(`/stories/${storyId}/items/${itemId}`, {
      method: 'DELETE',
    });
  }

  async updateStoryItemTimes(storyId: string, data: StoryItemBatchUpdate): Promise<void> {
    await this.request<void>(`/stories/${storyId}/items/times`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async reorderStoryItems(storyId: string, data: StoryItemReorder): Promise<StoryItemDetail[]> {
    return this.request<StoryItemDetail[]>(`/stories/${storyId}/items/reorder`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async moveStoryItem(
    storyId: string,
    itemId: string,
    data: StoryItemMove,
  ): Promise<StoryItemDetail> {
    return this.request<StoryItemDetail>(`/stories/${storyId}/items/${itemId}/move`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async trimStoryItem(
    storyId: string,
    itemId: string,
    data: StoryItemTrim,
  ): Promise<StoryItemDetail> {
    return this.request<StoryItemDetail>(`/stories/${storyId}/items/${itemId}/trim`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async updateStoryItemVolume(
    storyId: string,
    itemId: string,
    data: StoryItemVolumeUpdate,
  ): Promise<StoryItemDetail> {
    return this.request<StoryItemDetail>(`/stories/${storyId}/items/${itemId}/volume`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async splitStoryItem(
    storyId: string,
    itemId: string,
    data: StoryItemSplit,
  ): Promise<StoryItemDetail[]> {
    return this.request<StoryItemDetail[]>(`/stories/${storyId}/items/${itemId}/split`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async duplicateStoryItem(storyId: string, itemId: string): Promise<StoryItemDetail> {
    return this.request<StoryItemDetail>(`/stories/${storyId}/items/${itemId}/duplicate`, {
      method: 'POST',
    });
  }

  async setStoryItemVersion(
    storyId: string,
    itemId: string,
    data: StoryItemVersionUpdate,
  ): Promise<StoryItemDetail> {
    return this.request<StoryItemDetail>(`/stories/${storyId}/items/${itemId}/version`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async exportStoryAudio(storyId: string): Promise<Blob> {
    const url = `${this.getBaseUrl()}/stories/${storyId}/export-audio`;
    const response = await fetch(url);

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: response.statusText,
      }));
      throw new Error(formatErrorDetail(error.detail, `HTTP error! status: ${response.status}`));
    }

    return response.blob();
  }

  // Effects & Versions
  async getAvailableEffects(): Promise<AvailableEffectsResponse> {
    return this.request<AvailableEffectsResponse>('/effects/available');
  }

  async listEffectPresets(): Promise<EffectPresetResponse[]> {
    return this.request<EffectPresetResponse[]>('/effects/presets');
  }

  async createEffectPreset(data: EffectPresetCreate): Promise<EffectPresetResponse> {
    return this.request<EffectPresetResponse>('/effects/presets', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateEffectPreset(
    presetId: string,
    data: { name?: string; description?: string; effects_chain?: EffectConfig[] },
  ): Promise<EffectPresetResponse> {
    return this.request<EffectPresetResponse>(`/effects/presets/${presetId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteEffectPreset(presetId: string): Promise<void> {
    await this.request<void>(`/effects/presets/${presetId}`, {
      method: 'DELETE',
    });
  }

  async listGenerationVersions(generationId: string): Promise<GenerationVersionResponse[]> {
    return this.request<GenerationVersionResponse[]>(`/generations/${generationId}/versions`);
  }

  async applyEffectsToGeneration(
    generationId: string,
    data: ApplyEffectsRequest,
  ): Promise<GenerationVersionResponse> {
    return this.request<GenerationVersionResponse>(
      `/generations/${generationId}/versions/apply-effects`,
      {
        method: 'POST',
        body: JSON.stringify(data),
      },
    );
  }

  async setDefaultVersion(
    generationId: string,
    versionId: string,
  ): Promise<GenerationVersionResponse> {
    return this.request<GenerationVersionResponse>(
      `/generations/${generationId}/versions/${versionId}/set-default`,
      { method: 'PUT' },
    );
  }

  async deleteGenerationVersion(generationId: string, versionId: string): Promise<void> {
    await this.request<void>(`/generations/${generationId}/versions/${versionId}`, {
      method: 'DELETE',
    });
  }

  getVersionAudioUrl(versionId: string): string {
    return `${this.getBaseUrl()}/audio/version/${versionId}`;
  }

  async updateProfileEffects(
    profileId: string,
    effectsChain: EffectConfig[] | null,
  ): Promise<VoiceProfileResponse> {
    return this.request<VoiceProfileResponse>(`/profiles/${profileId}/effects`, {
      method: 'PUT',
      body: JSON.stringify({ effects_chain: effectsChain }),
    });
  }

  async previewEffects(generationId: string, effectsChain: EffectConfig[]): Promise<Blob> {
    const url = `${this.getBaseUrl()}/effects/preview/${generationId}`;
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ effects_chain: effectsChain }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: response.statusText,
      }));
      throw new Error(formatErrorDetail(error.detail, `HTTP error! status: ${response.status}`));
    }

    return response.blob();
  }

  // Cloud (backup & sync) — browser-based device login. startCloudLogin opens
  // the system browser server-side; the UI then polls getCloudStatus until the
  // backend completes the exchange and the link goes live.
  async getCloudStatus(): Promise<CloudStatus> {
    return this.request<CloudStatus>('/cloud/status');
  }

  async startCloudLogin(): Promise<CloudLoginStartResponse> {
    return this.request<CloudLoginStartResponse>('/cloud/login/start', { method: 'POST' });
  }

  async disconnectCloud(): Promise<CloudStatus> {
    return this.request<CloudStatus>('/cloud/disconnect', { method: 'POST' });
  }

  // ── Voice AI agent ──────────────────────────────────────────────────

  private async postForm<T>(endpoint: string, formData: FormData): Promise<T> {
    const response = await fetch(`${this.getBaseUrl()}${endpoint}`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(formatErrorDetail(error.detail, `HTTP error! status: ${response.status}`));
    }
    return response.json();
  }

  async listVoiceAgents(): Promise<VoiceAgent[]> {
    return this.request<VoiceAgent[]>('/agents');
  }

  async getVoiceAgent(agentId: string): Promise<VoiceAgent> {
    return this.request<VoiceAgent>(`/agents/${agentId}`);
  }

  async createVoiceAgent(data: VoiceAgentCreate): Promise<VoiceAgent> {
    return this.request<VoiceAgent>('/agents', { method: 'POST', body: JSON.stringify(data) });
  }

  async updateVoiceAgent(agentId: string, data: VoiceAgentUpdate): Promise<VoiceAgent> {
    return this.request<VoiceAgent>(`/agents/${agentId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteVoiceAgent(agentId: string): Promise<{ message: string }> {
    return this.request(`/agents/${agentId}`, { method: 'DELETE' });
  }

  async getVoiceAgentStats(agentId: string): Promise<VoiceAgentStats> {
    return this.request<VoiceAgentStats>(`/agents/${agentId}/stats`);
  }

  async getVoiceAgentAnalytics(agentId: string, days = 30): Promise<AnalyticsResponse> {
    return this.request<AnalyticsResponse>(`/agents/${agentId}/analytics?days=${days}`);
  }

  async startVoiceAgent(agentId: string): Promise<VoiceAgent> {
    return this.request<VoiceAgent>(`/agents/${agentId}/start`, { method: 'POST' });
  }

  async pauseVoiceAgent(agentId: string): Promise<VoiceAgent> {
    return this.request<VoiceAgent>(`/agents/${agentId}/pause`, { method: 'POST' });
  }

  async regenerateFillers(agentId: string): Promise<VoiceAgent> {
    return this.request<VoiceAgent>(`/agents/${agentId}/fillers/regenerate`, { method: 'POST' });
  }

  async listAgentVersions(agentId: string): Promise<VoiceAgentVersion[]> {
    return this.request<VoiceAgentVersion[]>(`/agents/${agentId}/versions`);
  }

  async restoreAgentVersion(agentId: string, versionId: string): Promise<VoiceAgent> {
    return this.request<VoiceAgent>(`/agents/${agentId}/versions/${versionId}/restore`, {
      method: 'POST',
    });
  }

  async simulateCall(agentId: string, data: SimulateRequest): Promise<VoiceCall> {
    return this.request<VoiceCall>(`/agents/${agentId}/simulate`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  getCallsCsvUrl(agentId: string, includeSimulations = false): string {
    return `${this.getBaseUrl()}/agents/${agentId}/export/calls.csv?include_simulations=${includeSimulations}`;
  }

  getContactsCsvUrl(agentId: string): string {
    return `${this.getBaseUrl()}/agents/${agentId}/export/contacts.csv`;
  }

  async listContacts(
    agentId: string,
    params: { status?: string; limit?: number; offset?: number } = {},
  ): Promise<ContactListResponse> {
    const qs = new URLSearchParams();
    if (params.status) qs.set('status', params.status);
    if (params.limit) qs.set('limit', String(params.limit));
    if (params.offset) qs.set('offset', String(params.offset));
    const suffix = qs.toString() ? `?${qs.toString()}` : '';
    return this.request<ContactListResponse>(`/agents/${agentId}/contacts${suffix}`);
  }

  async createContact(agentId: string, data: ContactCreate): Promise<Contact> {
    return this.request<Contact>(`/agents/${agentId}/contacts`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async bulkCreateContacts(
    agentId: string,
    contacts: ContactCreate[],
  ): Promise<ContactImportResult> {
    return this.request<ContactImportResult>(`/agents/${agentId}/contacts/bulk`, {
      method: 'POST',
      body: JSON.stringify({ contacts }),
    });
  }

  async importContactsCsv(agentId: string, file: File): Promise<ContactImportResult> {
    const formData = new FormData();
    formData.append('file', file);
    return this.postForm(`/agents/${agentId}/contacts/import`, formData);
  }

  async updateContact(contactId: string, data: ContactUpdate): Promise<Contact> {
    return this.request<Contact>(`/contacts/${contactId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteContact(contactId: string): Promise<{ message: string }> {
    return this.request(`/contacts/${contactId}`, { method: 'DELETE' });
  }

  async listKnowledge(agentId: string): Promise<KnowledgeArticle[]> {
    return this.request<KnowledgeArticle[]>(`/agents/${agentId}/knowledge`);
  }

  async createKnowledge(agentId: string, data: KnowledgeArticleCreate): Promise<KnowledgeArticle> {
    return this.request<KnowledgeArticle>(`/agents/${agentId}/knowledge`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async importKnowledgeUrl(
    agentId: string,
    url: string,
    tags?: string[],
  ): Promise<KnowledgeArticle[]> {
    return this.request<KnowledgeArticle[]>(`/agents/${agentId}/knowledge/import-url`, {
      method: 'POST',
      body: JSON.stringify({ url, tags: tags ?? null }),
    });
  }

  async importKnowledgeFile(
    agentId: string,
    file: File,
    tags?: string,
  ): Promise<KnowledgeArticle[]> {
    const formData = new FormData();
    formData.append('file', file);
    if (tags) formData.append('tags', tags);
    return this.postForm(`/agents/${agentId}/knowledge/import-file`, formData);
  }

  async searchKnowledge(agentId: string, q: string): Promise<KnowledgeSearchResult[]> {
    return this.request<KnowledgeSearchResult[]>(
      `/agents/${agentId}/knowledge/search?q=${encodeURIComponent(q)}`,
    );
  }

  async updateKnowledge(
    articleId: string,
    data: Partial<KnowledgeArticleCreate>,
  ): Promise<KnowledgeArticle> {
    return this.request<KnowledgeArticle>(`/knowledge/${articleId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteKnowledge(articleId: string): Promise<{ message: string }> {
    return this.request(`/knowledge/${articleId}`, { method: 'DELETE' });
  }

  async listTools(agentId: string): Promise<AgentTool[]> {
    return this.request<AgentTool[]>(`/agents/${agentId}/tools`);
  }

  async createTool(agentId: string, data: AgentToolCreate): Promise<AgentTool> {
    return this.request<AgentTool>(`/agents/${agentId}/tools`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateTool(toolId: string, data: Partial<AgentToolCreate>): Promise<AgentTool> {
    return this.request<AgentTool>(`/tools/${toolId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteTool(toolId: string): Promise<{ message: string }> {
    return this.request(`/tools/${toolId}`, { method: 'DELETE' });
  }

  async listCalls(
    agentId: string,
    params: { status?: string; limit?: number; offset?: number; includeSimulations?: boolean } = {},
  ): Promise<CallListResponse> {
    const qs = new URLSearchParams();
    if (params.status) qs.set('status', params.status);
    if (params.limit) qs.set('limit', String(params.limit));
    if (params.offset) qs.set('offset', String(params.offset));
    if (params.includeSimulations === false) qs.set('include_simulations', 'false');
    const suffix = qs.toString() ? `?${qs.toString()}` : '';
    return this.request<CallListResponse>(`/agents/${agentId}/calls${suffix}`);
  }

  async startNextCall(
    agentId: string,
    opts: { contactId?: string; variant?: string; clientPlays?: boolean } = {},
  ): Promise<AgentTurnResponse> {
    const qs = new URLSearchParams();
    if (opts.contactId) qs.set('contact_id', opts.contactId);
    if (opts.variant) qs.set('variant', opts.variant);
    if (opts.clientPlays) qs.set('client_plays', 'true');
    const suffix = qs.toString() ? `?${qs.toString()}` : '';
    return this.request<AgentTurnResponse>(`/agents/${agentId}/calls/next${suffix}`, {
      method: 'POST',
    });
  }

  async startInboundCall(
    agentId: string,
    data: { phone: string; name?: string | null },
    clientPlays = false,
  ): Promise<AgentTurnResponse> {
    const suffix = clientPlays ? '?client_plays=true' : '';
    return this.request<AgentTurnResponse>(`/agents/${agentId}/calls/inbound${suffix}`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getCall(callId: string): Promise<VoiceCall> {
    return this.request<VoiceCall>(`/calls/${callId}`);
  }

  getCallEventsUrl(callId: string): string {
    return `${this.getBaseUrl()}/calls/${callId}/events`;
  }

  getCallTranscriptUrl(callId: string): string {
    return `${this.getBaseUrl()}/calls/${callId}/transcript.txt`;
  }

  async sendCustomerTurn(
    callId: string,
    text: string,
    clientPlays = false,
  ): Promise<AgentTurnResponse> {
    const suffix = clientPlays ? '?client_plays=true' : '';
    return this.request<AgentTurnResponse>(`/calls/${callId}/turn${suffix}`, {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
  }

  async sendCustomerTurnAudio(
    callId: string,
    blob: Blob,
    language?: string,
    clientPlays = false,
  ): Promise<AgentTurnResponse> {
    const formData = new FormData();
    formData.append('file', blob, 'turn.wav');
    if (language) formData.append('language', language);
    const suffix = clientPlays ? '?client_plays=true' : '';
    return this.postForm(`/calls/${callId}/turn/audio${suffix}`, formData);
  }

  async interruptCall(
    callId: string,
    turnId?: string,
  ): Promise<{ interrupted_turn_id: string | null }> {
    return this.request(`/calls/${callId}/interrupt`, {
      method: 'POST',
      body: JSON.stringify({ turn_id: turnId ?? null }),
    });
  }

  async agentSay(callId: string, text: string, clientPlays = false): Promise<AgentTurnResponse> {
    const suffix = clientPlays ? '?client_plays=true' : '';
    return this.request<AgentTurnResponse>(`/calls/${callId}/agent_say${suffix}`, {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
  }

  async setCallAiPaused(callId: string, paused: boolean): Promise<VoiceCall> {
    return this.request<VoiceCall>(`/calls/${callId}/ai/${paused ? 'pause' : 'resume'}`, {
      method: 'POST',
    });
  }

  async endCall(callId: string, outcome: string, summary?: string): Promise<VoiceCall> {
    return this.request<VoiceCall>(`/calls/${callId}/end`, {
      method: 'POST',
      body: JSON.stringify({ outcome, summary: summary ?? null }),
    });
  }

  async listAppointments(agentId: string, upcoming = false): Promise<Appointment[]> {
    return this.request<Appointment[]>(`/agents/${agentId}/appointments?upcoming=${upcoming}`);
  }

  async updateAppointment(
    appointmentId: string,
    data: { status?: string; notes?: string; starts_at?: string; ends_at?: string },
  ): Promise<Appointment> {
    return this.request<Appointment>(`/appointments/${appointmentId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  getAppointmentIcsUrl(appointmentId: string): string {
    return `${this.getBaseUrl()}/appointments/${appointmentId}.ics`;
  }

  async listAgentMessages(agentId: string): Promise<AgentMessage[]> {
    return this.request<AgentMessage[]>(`/agents/${agentId}/messages`);
  }

  async listWebhookDeliveries(agentId: string): Promise<WebhookDelivery[]> {
    return this.request<WebhookDelivery[]>(`/agents/${agentId}/webhook-deliveries`);
  }

  async testWebhook(agentId: string): Promise<WebhookDelivery> {
    return this.request<WebhookDelivery>(`/agents/${agentId}/webhook/test`, { method: 'POST' });
  }

  async listTickets(
    params: { agentId?: string; status?: string; limit?: number } = {},
  ): Promise<TicketListResponse> {
    const qs = new URLSearchParams();
    if (params.agentId) qs.set('agent_id', params.agentId);
    if (params.status) qs.set('status', params.status);
    if (params.limit) qs.set('limit', String(params.limit));
    const suffix = qs.toString() ? `?${qs.toString()}` : '';
    return this.request<TicketListResponse>(`/tickets${suffix}`);
  }

  async updateTicket(
    ticketId: string,
    data: { status?: string; priority?: string; description?: string },
  ): Promise<Ticket> {
    return this.request<Ticket>(`/tickets/${ticketId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async listDoNotCall(): Promise<DoNotCallEntry[]> {
    return this.request<DoNotCallEntry[]>('/dnc');
  }

  async addDoNotCall(phone: string, reason?: string): Promise<DoNotCallEntry> {
    return this.request<DoNotCallEntry>('/dnc', {
      method: 'POST',
      body: JSON.stringify({ phone, reason: reason ?? null }),
    });
  }

  async importDoNotCall(file: File): Promise<DoNotCallImportResult> {
    const formData = new FormData();
    formData.append('file', file);
    return this.postForm('/dnc/import', formData);
  }

  async removeDoNotCall(phone: string): Promise<{ message: string }> {
    return this.request(`/dnc/${encodeURIComponent(phone)}`, { method: 'DELETE' });
  }
}

export const apiClient = new ApiClient();
