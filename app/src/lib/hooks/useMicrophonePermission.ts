import { emit } from '@tauri-apps/api/event';
import { create } from 'zustand';

export const MICROPHONE_BLOCKED_EVENT = 'system:microphone-blocked';

const PERMISSION_ERROR_NAMES = new Set(['NotAllowedError', 'PermissionDeniedError']);

export function isMicrophonePermissionError(err: unknown): boolean {
  return (
    typeof err === 'object' &&
    err !== null &&
    PERMISSION_ERROR_NAMES.has((err as { name?: unknown }).name as string)
  );
}

export async function queryMicrophonePermission(): Promise<PermissionState | null> {
  try {
    const status = await navigator.permissions?.query({ name: 'microphone' as PermissionName });
    return status?.state ?? null;
  } catch {
    return null;
  }
}

export async function isMicrophoneDenied(err: unknown): Promise<boolean> {
  return isMicrophonePermissionError(err) || (await queryMicrophonePermission()) === 'denied';
}

interface MicrophonePermissionStore {
  blocked: boolean;
  setBlocked: (blocked: boolean) => void;
}

export const useMicrophonePermissionStore = create<MicrophonePermissionStore>((set) => ({
  blocked: false,
  setBlocked: (blocked) => set({ blocked }),
}));

/**
 * Opens the shared recovery dialog. In Tauri the event also reaches the main
 * window when the denial happened in the floating dictate window, which has
 * no room for the dialog, and lets the native side bring the main window
 * forward or open the OS settings when the main window is closed.
 */
export function reportMicrophoneBlocked(isTauri: boolean) {
  useMicrophonePermissionStore.getState().setBlocked(true);
  if (isTauri) emit(MICROPHONE_BLOCKED_EVENT).catch(() => {});
}

export async function requestMicrophoneAccess(): Promise<boolean> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stream.getTracks().forEach((track) => {
      track.stop();
    });
    return true;
  } catch {
    return false;
  }
}
