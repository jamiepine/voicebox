// Shared app-level type aliases.
// Keep these aliases in sync with the generated API contract types.
import type {
  GenerationResponse,
  VoiceProfileResponse,
} from '@/lib/api/types';

export type VoiceProfile = VoiceProfileResponse;

export type Generation = GenerationResponse;

export interface ServerConfig {
  url: string;
  isRemote: boolean;
  isRunning: boolean;
}
