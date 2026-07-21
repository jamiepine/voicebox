import { invoke } from '@tauri-apps/api/core';
import { emit, listen } from '@tauri-apps/api/event';
import { useEffect, useRef } from 'react';
import { useDictationReadiness } from '@/lib/hooks/useDictationReadiness';
import { useCaptureSettings } from '@/lib/hooks/useSettings';
import { usePlatform } from '@/platform/PlatformContext';

/**
 * Spawn (or quiet) the global hotkey monitor based on the saved
 * `capture_settings.hotkey_enabled` flag and the recording readiness gates,
 * and keep its bindings in sync with the user's chord choices.
 *
 * Boot sequence:
 *  - hotkey_enabled = false OR a recording gate is missing → call
 *    `disable_hotkey` (no-op if monitor was never spawned). Crucially, we do
 *    *not* call `enable_hotkey` in this state, so the macOS Input Monitoring
 *    TCC prompt is never triggered for users who haven't opted in, AND the
 *    chord physically can't fire when models aren't downloaded — preventing
 *    the "stuck pill" failure mode where dictation triggers but has nowhere
 *    to land.
 *  - hotkey_enabled = true AND recording gates green → call `enable_hotkey` with
 *    the saved chords. This creates the CGEventTap and triggers the TCC
 *    prompt on first opt-in. Re-runs whenever a gate flips green (e.g. the
 *    user finishes downloading Whisper in another tab) so the chord
 *    auto-arms without making the user toggle off/on.
 *
 * Call once from the main app shell.
 */
export function useChordSync() {
  const platform = usePlatform();
  const { settings } = useCaptureSettings();
  const { canRecord } = useDictationReadiness();
  const enabled = settings?.hotkey_enabled;
  const keepMicWarm = settings?.keep_mic_warm;
  const pushKeys = settings?.chord_push_to_talk_keys;
  const toggleKeys = settings?.chord_toggle_to_talk_keys;

  // Latest warm state, so the dictate window's mount-time request can be
  // answered even between the dep-driven emits below.
  const shouldWarmRef = useRef(false);

  // The floating dictate window holds the mic warm ahead of the first chord to
  // avoid clipping, but it's a separate webview with no view of settings. Mirror
  // the decision to it: warm only when dictation is armed AND the user enabled
  // "keep microphone ready". Gating here is what stops the always-mounted pill
  // from opening the mic — or prompting for access — when the user hasn't asked.
  useEffect(() => {
    if (!platform.metadata.isTauri) return;
    const unlisten = listen('dictate:warm-request', () => {
      emit('dictate:warm', shouldWarmRef.current).catch(() => {});
    });
    return () => {
      unlisten.then((fn) => fn()).catch(() => {});
    };
  }, [platform.metadata.isTauri]);

  useEffect(() => {
    if (!platform.metadata.isTauri) return;
    if (enabled === undefined || !pushKeys || !toggleKeys) return;
    const shouldArm = enabled && canRecord;
    const shouldWarm = shouldArm && (keepMicWarm ?? false);
    shouldWarmRef.current = shouldWarm;
    const command = shouldArm ? 'enable_hotkey' : 'disable_hotkey';
    const args = shouldArm ? { pushToTalk: pushKeys, toggleToTalk: toggleKeys } : {};
    invoke(command, args).catch((err) => {
      console.warn(`[chord-sync] ${command} failed:`, err);
    });
    emit('dictate:warm', shouldWarm).catch(() => {});
  }, [
    platform.metadata.isTauri,
    enabled,
    keepMicWarm,
    canRecord,
    // Stringify so a referentially-new array with the same content
    // doesn't fire a redundant invoke on every settings refetch.
    pushKeys?.join(','),
    toggleKeys?.join(','),
  ]);
}
