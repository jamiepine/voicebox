import { useState, useRef, useCallback, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { isTauri } from '@/lib/tauri';

interface UseSystemAudioCaptureOptions {
  maxDurationSeconds?: number;
  onRecordingComplete?: (blob: Blob, duration?: number) => void;
}

/**
 * Hook for native system audio capture using Tauri commands.
 * Uses ScreenCaptureKit on macOS and WASAPI loopback on Windows.
 */
export function useSystemAudioCapture({
  maxDurationSeconds = 30,
  onRecordingComplete,
}: UseSystemAudioCaptureOptions = {}) {
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isSupported, setIsSupported] = useState(false);
  const timerRef = useRef<number | null>(null);
  const startTimeRef = useRef<number | null>(null);
  const stopRecordingRef = useRef<(() => Promise<void>) | null>(null);
  const isRecordingRef = useRef(false);

  // Check if system audio capture is supported
  useEffect(() => {
    if (!isTauri()) {
      setIsSupported(false);
      return;
    }

    invoke<boolean>('is_system_audio_supported')
      .then((supported) => {
        setIsSupported(supported);
      })
      .catch(() => {
        setIsSupported(false);
      });
  }, []);

  const startRecording = useCallback(async () => {
    console.log('[useSystemAudioCapture] startRecording called');
    if (!isTauri()) {
      const errorMsg = 'System audio capture is only available in the desktop app.';
      setError(errorMsg);
      return;
    }

    if (!isSupported) {
      const errorMsg = 'System audio capture is not supported on this platform.';
      setError(errorMsg);
      return;
    }

    try {
      console.log('[useSystemAudioCapture] Starting capture...');
      setError(null);
      setDuration(0);

      // Start native capture
      console.log('[useSystemAudioCapture] Calling invoke...');
      await invoke('start_system_audio_capture', {
        maxDurationSecs: maxDurationSeconds,
      });
      console.log('[useSystemAudioCapture] Invoke completed, starting timer');

      setIsRecording(true);
      isRecordingRef.current = true;
      startTimeRef.current = Date.now();

      // Start timer
      timerRef.current = window.setInterval(() => {
        if (startTimeRef.current) {
          const elapsed = (Date.now() - startTimeRef.current) / 1000;
          setDuration(elapsed);

          // Auto-stop at max duration
          if (elapsed >= maxDurationSeconds && stopRecordingRef.current) {
            console.log('[useSystemAudioCapture] Max duration reached, auto-stopping');
            void stopRecordingRef.current();
          }
        }
      }, 100);
      console.log('[useSystemAudioCapture] Timer started');
    } catch (err) {
      console.error('[useSystemAudioCapture] Error starting recording:', err);
      const errorMessage =
        err instanceof Error
          ? err.message
          : 'Failed to start system audio capture. Please check permissions.';
      setError(errorMessage);
      setIsRecording(false);
    }
  }, [maxDurationSeconds, isSupported]);

  const stopRecording = useCallback(async () => {
    if (!isRecording || !isTauri()) {
      return;
    }

    try {
      setIsRecording(false);
      isRecordingRef.current = false;

      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }

      // Stop capture and get base64 WAV data
      const base64Data = await invoke<string>('stop_system_audio_capture');

      // Convert base64 to Blob
      const binaryString = atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      const blob = new Blob([bytes], { type: 'audio/wav' });
      // Pass the actual recorded duration
      const recordedDuration = startTimeRef.current 
        ? (Date.now() - startTimeRef.current) / 1000 
        : undefined;
      onRecordingComplete?.(blob, recordedDuration);
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : 'Failed to stop system audio capture.';
      setError(errorMessage);
    }
  }, [isRecording, onRecordingComplete]);

  // Store stopRecording in ref for use in timer
  useEffect(() => {
    stopRecordingRef.current = stopRecording;
  }, [stopRecording]);

  const cancelRecording = useCallback(async () => {
    if (isRecordingRef.current) {
      await stopRecording();
    }

    setIsRecording(false);
    isRecordingRef.current = false;
    setDuration(0);

    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, [stopRecording]);

  // Cleanup on unmount only
  useEffect(() => {
    return () => {
      console.log('[useSystemAudioCapture] Component unmounting, cleaning up');
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      // Cancel recording on unmount if still recording
      if (isRecordingRef.current && isTauri()) {
        console.log('[useSystemAudioCapture] Canceling recording on unmount');
        // Call stop directly without the callback to avoid stale closure
        invoke('stop_system_audio_capture').catch((err) => {
          console.error('[useSystemAudioCapture] Error stopping on unmount:', err);
        });
      }
    };
    // biome-ignore lint/correctness/useExhaustiveDependencies: Only run on unmount
  }, []);

  return {
    isRecording,
    duration,
    error,
    isSupported,
    startRecording,
    stopRecording,
    cancelRecording,
  };
}
