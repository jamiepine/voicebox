import { invoke } from '@tauri-apps/api/core';
import type { PlatformAudio, AudioDevice } from '@/platform/types';

export const tauriAudio: PlatformAudio = {
  isSystemAudioSupported(): boolean {
    // This will be checked dynamically via invoke
    return true; // Tauri supports it, but actual support depends on platform
  },

  async startSystemAudioCapture(maxDurationSecs: number): Promise<void> {
    await invoke('start_system_audio_capture', {
      maxDurationSecs,
    });
  },

  async stopSystemAudioCapture(): Promise<Blob> {
    const base64Data = await invoke<string>('stop_system_audio_capture');

    // Convert base64 to Blob
    const binaryString = atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    return new Blob([bytes], { type: 'audio/wav' });
  },

  async listOutputDevices(): Promise<AudioDevice[]> {
    return await invoke<AudioDevice[]>('list_audio_output_devices');
  },

  async listInputDevices(): Promise<AudioDevice[]> {
    return await invoke<AudioDevice[]>('list_audio_input_devices');
  },

  async playToDevices(audioData: Uint8Array, deviceIds: string[]): Promise<void> {
    await invoke('play_audio_to_devices', {
      audioData: Array.from(audioData),
      deviceIds,
    });
  },

  stopPlayback(): void {
    invoke('stop_audio_playback').catch((error) => {
      console.error('Failed to stop audio playback:', error);
    });
  },
};
