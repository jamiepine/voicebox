/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
import type { Body_add_profile_sample_profiles__profile_id__samples_post } from '../models/Body_add_profile_sample_profiles__profile_id__samples_post';
import type { Body_transcribe_audio_transcribe_post } from '../models/Body_transcribe_audio_transcribe_post';
import type { GenerationRequest } from '../models/GenerationRequest';
import type { GenerationResponse } from '../models/GenerationResponse';
import type { HealthResponse } from '../models/HealthResponse';
import type { HistoryListResponse } from '../models/HistoryListResponse';
import type { HistoryResponse } from '../models/HistoryResponse';
import type { ModelDownloadRequest } from '../models/ModelDownloadRequest';
import type { ModelStatusListResponse } from '../models/ModelStatusListResponse';
import type { ProfileSampleResponse } from '../models/ProfileSampleResponse';
import type { TranscriptionResponse } from '../models/TranscriptionResponse';
import type { VoiceProfileCreate } from '../models/VoiceProfileCreate';
import type { VoiceProfileResponse } from '../models/VoiceProfileResponse';
export class DefaultService {
  /**
   * Root
   * Root endpoint.
   * @returns any Successful Response
   * @throws ApiError
   */
  public static rootGet(): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/',
    });
  }
  /**
   * Health
   * Health check endpoint.
   * @returns HealthResponse Successful Response
   * @throws ApiError
   */
  public static healthHealthGet(): CancelablePromise<HealthResponse> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/health',
    });
  }
  /**
   * List Profiles
   * List all voice profiles.
   * @returns VoiceProfileResponse Successful Response
   * @throws ApiError
   */
  public static listProfilesProfilesGet(): CancelablePromise<Array<VoiceProfileResponse>> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/profiles',
    });
  }
  /**
   * Create Profile
   * Create a new voice profile.
   * @returns VoiceProfileResponse Successful Response
   * @throws ApiError
   */
  public static createProfileProfilesPost({
    requestBody,
  }: {
    requestBody: VoiceProfileCreate;
  }): CancelablePromise<VoiceProfileResponse> {
    return __request(OpenAPI, {
      method: 'POST',
      url: '/profiles',
      body: requestBody,
      mediaType: 'application/json',
      errors: {
        422: `Validation Error`,
      },
    });
  }
  /**
   * Get Profile
   * Get a voice profile by ID.
   * @returns VoiceProfileResponse Successful Response
   * @throws ApiError
   */
  public static getProfileProfilesProfileIdGet({
    profileId,
  }: {
    profileId: string;
  }): CancelablePromise<VoiceProfileResponse> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/profiles/{profile_id}',
      path: {
        profile_id: profileId,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }
  /**
   * Update Profile
   * Update a voice profile.
   * @returns VoiceProfileResponse Successful Response
   * @throws ApiError
   */
  public static updateProfileProfilesProfileIdPut({
    profileId,
    requestBody,
  }: {
    profileId: string;
    requestBody: VoiceProfileCreate;
  }): CancelablePromise<VoiceProfileResponse> {
    return __request(OpenAPI, {
      method: 'PUT',
      url: '/profiles/{profile_id}',
      path: {
        profile_id: profileId,
      },
      body: requestBody,
      mediaType: 'application/json',
      errors: {
        422: `Validation Error`,
      },
    });
  }
  /**
   * Delete Profile
   * Delete a voice profile.
   * @returns any Successful Response
   * @throws ApiError
   */
  public static deleteProfileProfilesProfileIdDelete({
    profileId,
  }: {
    profileId: string;
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'DELETE',
      url: '/profiles/{profile_id}',
      path: {
        profile_id: profileId,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }
  /**
   * Add Profile Sample
   * Add a sample to a voice profile.
   * @returns ProfileSampleResponse Successful Response
   * @throws ApiError
   */
  public static addProfileSampleProfilesProfileIdSamplesPost({
    profileId,
    formData,
  }: {
    profileId: string;
    formData: Body_add_profile_sample_profiles__profile_id__samples_post;
  }): CancelablePromise<ProfileSampleResponse> {
    return __request(OpenAPI, {
      method: 'POST',
      url: '/profiles/{profile_id}/samples',
      path: {
        profile_id: profileId,
      },
      formData: formData,
      mediaType: 'multipart/form-data',
      errors: {
        422: `Validation Error`,
      },
    });
  }
  /**
   * Get Profile Samples
   * Get all samples for a profile.
   * @returns ProfileSampleResponse Successful Response
   * @throws ApiError
   */
  public static getProfileSamplesProfilesProfileIdSamplesGet({
    profileId,
  }: {
    profileId: string;
  }): CancelablePromise<Array<ProfileSampleResponse>> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/profiles/{profile_id}/samples',
      path: {
        profile_id: profileId,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }
  /**
   * Delete Profile Sample
   * Delete a profile sample.
   * @returns any Successful Response
   * @throws ApiError
   */
  public static deleteProfileSampleProfilesSamplesSampleIdDelete({
    sampleId,
  }: {
    sampleId: string;
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'DELETE',
      url: '/profiles/samples/{sample_id}',
      path: {
        sample_id: sampleId,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }
  /**
   * Generate Speech
   * Generate speech from text using a voice profile.
   * @returns GenerationResponse Successful Response
   * @throws ApiError
   */
  public static generateSpeechGeneratePost({
    requestBody,
  }: {
    requestBody: GenerationRequest;
  }): CancelablePromise<GenerationResponse> {
    return __request(OpenAPI, {
      method: 'POST',
      url: '/generate',
      body: requestBody,
      mediaType: 'application/json',
      errors: {
        422: `Validation Error`,
      },
    });
  }
  /**
   * List History
   * List generation history with optional filters.
   * @returns HistoryListResponse Successful Response
   * @throws ApiError
   */
  public static listHistoryHistoryGet({
    profileId,
    search,
    limit = 50,
    offset,
  }: {
    profileId?: string | null;
    search?: string | null;
    limit?: number;
    offset?: number;
  }): CancelablePromise<HistoryListResponse> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/history',
      query: {
        profile_id: profileId,
        search: search,
        limit: limit,
        offset: offset,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }
  /**
   * Get Generation
   * Get a generation by ID.
   * @returns HistoryResponse Successful Response
   * @throws ApiError
   */
  public static getGenerationHistoryGenerationIdGet({
    generationId,
  }: {
    generationId: string;
  }): CancelablePromise<HistoryResponse> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/history/{generation_id}',
      path: {
        generation_id: generationId,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }
  /**
   * Delete Generation
   * Delete a generation.
   * @returns any Successful Response
   * @throws ApiError
   */
  public static deleteGenerationHistoryGenerationIdDelete({
    generationId,
  }: {
    generationId: string;
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'DELETE',
      url: '/history/{generation_id}',
      path: {
        generation_id: generationId,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }
  /**
   * Get Stats
   * Get generation statistics.
   * @returns any Successful Response
   * @throws ApiError
   */
  public static getStatsHistoryStatsGet(): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/history/stats',
    });
  }
  /**
   * Transcribe Audio
   * Transcribe audio file to text.
   * @returns TranscriptionResponse Successful Response
   * @throws ApiError
   */
  public static transcribeAudioTranscribePost({
    formData,
  }: {
    formData: Body_transcribe_audio_transcribe_post;
  }): CancelablePromise<TranscriptionResponse> {
    return __request(OpenAPI, {
      method: 'POST',
      url: '/transcribe',
      formData: formData,
      mediaType: 'multipart/form-data',
      errors: {
        422: `Validation Error`,
      },
    });
  }
  /**
   * Get Audio
   * Serve generated audio file.
   * @returns any Successful Response
   * @throws ApiError
   */
  public static getAudioAudioGenerationIdGet({
    generationId,
  }: {
    generationId: string;
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/audio/{generation_id}',
      path: {
        generation_id: generationId,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }
  /**
   * Load Model
   * Manually load TTS model.
   * @returns any Successful Response
   * @throws ApiError
   */
  public static loadModelModelsLoadPost({
    modelSize = '1.7B',
  }: {
    modelSize?: string;
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'POST',
      url: '/models/load',
      query: {
        model_size: modelSize,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }
  /**
   * Unload Model
   * Unload TTS model to free memory.
   * @returns any Successful Response
   * @throws ApiError
   */
  public static unloadModelModelsUnloadPost(): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'POST',
      url: '/models/unload',
    });
  }
  /**
   * Get Model Progress
   * Get model download progress via Server-Sent Events.
   * @returns any Successful Response
   * @throws ApiError
   */
  public static getModelProgressModelsProgressModelNameGet({
    modelName,
  }: {
    modelName: string;
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/models/progress/{model_name}',
      path: {
        model_name: modelName,
      },
      errors: {
        422: `Validation Error`,
      },
    });
  }
  /**
   * Get Model Status
   * Get status of all available models.
   * @returns ModelStatusListResponse Successful Response
   * @throws ApiError
   */
  public static getModelStatusModelsStatusGet(): CancelablePromise<ModelStatusListResponse> {
    return __request(OpenAPI, {
      method: 'GET',
      url: '/models/status',
    });
  }
  /**
   * Trigger Model Download
   * Trigger download of a specific model.
   * @returns any Successful Response
   * @throws ApiError
   */
  public static triggerModelDownloadModelsDownloadPost({
    requestBody,
  }: {
    requestBody: ModelDownloadRequest;
  }): CancelablePromise<any> {
    return __request(OpenAPI, {
      method: 'POST',
      url: '/models/download',
      body: requestBody,
      mediaType: 'application/json',
      errors: {
        422: `Validation Error`,
      },
    });
  }
}
