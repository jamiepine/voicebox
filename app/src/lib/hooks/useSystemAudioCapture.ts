import { useCallback, useEffect, useRef, useState } from 'react';
import { usePlatform } from '@/platform/PlatformContext';

interface UseSystemAudioCaptureOptions {
  maxDurationSeconds?: number;
  onRecordingComplete?: (blob: Blob, duration?: number) => void;
}

/**
 * Hook for native system audio capture using Tauri commands.
 * Uses ScreenCaptureKit on macOS and WASAPI loopback on Windows.
 */
export function useSystemAudioCapture({
  maxDurationSeconds = 29,
  onRecordingComplete,
}: UseSystemAudioCaptureOptions = {}) {
  const platform = usePlatform();
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
    let isActive = true;

    void platform.audio
      .isSystemAudioSupported()
      .then((supported) => {
        if (isActive) {
          setIsSupported(supported);
        }
      })
      .catch(() => {
        if (isActive) {
          setIsSupported(false);
        }
      });

    return () => {
      isActive = false;
    };
  }, [platform]);

  const startRecording = useCallback(async () => {
    if (!platform.metadata.isTauri) {
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
      setError(null);
      setDuration(0);

      // Start native capture
      await platform.audio.startSystemAudioCapture(maxDurationSeconds);

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
            void stopRecordingRef.current();
          }
        }
      }, 100);
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : 'Failed to start system audio capture. Please check permissions.';
      setError(errorMessage);
      setIsRecording(false);
    }
  }, [maxDurationSeconds, isSupported, platform]);

  const stopRecording = useCallback(async () => {
    if (!isRecording || !platform.metadata.isTauri) {
      return;
    }

    try {
      setIsRecording(false);
      isRecordingRef.current = false;

      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }

      // Stop capture and get Blob
      const blob = await platform.audio.stopSystemAudioCapture();

      // Pass the actual recorded duration
      const recordedDuration = startTimeRef.current
        ? (Date.now() - startTimeRef.current) / 1000
        : undefined;
      onRecordingComplete?.(blob, recordedDuration);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : 'Failed to stop system audio capture.';
      setError(errorMessage);
    }
  }, [isRecording, onRecordingComplete, platform]);

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
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      // Cancel recording on unmount if still recording
      if (isRecordingRef.current && platform.metadata.isTauri) {
        // Call stop directly without the callback to avoid stale closure
        platform.audio.stopSystemAudioCapture().catch((err) => {
          console.error('Error stopping audio capture on unmount:', err);
        });
      }
    };
    // biome-ignore lint/correctness/useExhaustiveDependencies: Only run on unmount
  }, [platform]);

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
