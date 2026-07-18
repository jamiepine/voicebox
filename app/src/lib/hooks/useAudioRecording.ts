import { useCallback, useEffect, useRef, useState } from 'react';
import { usePlatform } from '@/platform/PlatformContext';
import { convertToWav } from '@/lib/utils/audio';

interface UseAudioRecordingOptions {
  maxDurationSeconds?: number;
  onRecordingComplete?: (blob: Blob, duration?: number) => void;
  /**
   * Keep the microphone ``MediaStream`` open between recordings instead of
   * tearing it down on every stop. This is what removes the "first words get
   * clipped" problem on push-to-talk dictation: ``getUserMedia`` on macOS can
   * take several hundred ms — up to a second cold — to hand back a stream, and
   * ``MediaRecorder`` only starts capturing *after* it resolves, so everything
   * spoken in that window is lost. With a warm stream already open, the next
   * ``startRecording`` skips ``getUserMedia`` entirely and ``MediaRecorder``
   * captures from the first frame.
   *
   * Off by default so the voice-clone sample recorders (which record once and
   * should release the device immediately) keep their existing behavior; only
   * the dictation session opts in. When on, the stream is released after
   * ``WARM_IDLE_RELEASE_MS`` of inactivity so the OS mic indicator doesn't stay
   * lit forever when the user isn't dictating.
   */
  keepWarm?: boolean;
}

// Audio constraints for capture. Kept identical to the previous inline value so
// this change is purely about *when* the stream is opened, not *how*.
const AUDIO_CONSTRAINTS: MediaTrackConstraints = {
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
};

// How long a warm stream lingers after the last recording before it's released.
// Long enough that back-to-back dictations reuse the same warm stream (no
// clipping), short enough that the mic indicator clears soon after the user
// stops. A follow-up could re-warm just-in-time on a chord-arming event from
// Rust to also cover the first dictation after this window elapses.
const WARM_IDLE_RELEASE_MS = 60_000;

const streamHasLiveAudio = (stream: MediaStream | null): stream is MediaStream =>
  !!stream && stream.getAudioTracks().some((t) => t.readyState === 'live');

export function useAudioRecording({
  maxDurationSeconds,
  onRecordingComplete,
  keepWarm = false,
}: UseAudioRecordingOptions = {}) {
  const platform = usePlatform();
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  // The stream currently backing the MediaRecorder. When ``keepWarm`` is set
  // this is the same object as ``warmStreamRef`` and is *not* torn down on
  // stop; otherwise it's stopped as soon as the recording completes.
  const streamRef = useRef<MediaStream | null>(null);
  // Persistent pre-opened stream reused across recordings when ``keepWarm``.
  const warmStreamRef = useRef<MediaStream | null>(null);
  const idleReleaseTimerRef = useRef<number | null>(null);
  const timerRef = useRef<number | null>(null);
  const startTimeRef = useRef<number | null>(null);
  const cancelledRef = useRef<boolean>(false);

  const clearIdleRelease = useCallback(() => {
    if (idleReleaseTimerRef.current !== null) {
      window.clearTimeout(idleReleaseTimerRef.current);
      idleReleaseTimerRef.current = null;
    }
  }, []);

  const releaseWarmStream = useCallback(() => {
    clearIdleRelease();
    warmStreamRef.current?.getTracks().forEach((track) => {
      track.stop();
    });
    warmStreamRef.current = null;
  }, [clearIdleRelease]);

  const scheduleIdleRelease = useCallback(() => {
    if (!keepWarm) return;
    clearIdleRelease();
    idleReleaseTimerRef.current = window.setTimeout(() => {
      idleReleaseTimerRef.current = null;
      releaseWarmStream();
    }, WARM_IDLE_RELEASE_MS);
  }, [keepWarm, clearIdleRelease, releaseWarmStream]);

  // Assert that getUserMedia is reachable, mirroring the previous inline guard
  // (Tauri webviews occasionally expose ``navigator.mediaDevices`` a beat late).
  const assertMediaDevices = useCallback(async () => {
    if (typeof navigator === 'undefined') {
      throw new Error('Navigator API is not available. This might be a Tauri configuration issue.');
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      await new Promise((resolve) => setTimeout(resolve, 100));
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
          platform.metadata.isTauri
            ? 'Microphone access is not available. Please ensure:\n1. The app has microphone permissions in System Settings (macOS: System Settings > Privacy & Security > Microphone)\n2. You restart the app after granting permissions\n3. You are using Tauri v2 with a webview that supports getUserMedia'
            : 'Microphone access is not available. Please ensure you are using a secure context (HTTPS or localhost) and that your browser has microphone permissions enabled.',
        );
      }
    }
  }, [platform.metadata.isTauri]);

  // Return a live capture stream, reusing the warm one when available so the
  // hot path (chord-down → record) never waits on getUserMedia.
  const acquireStream = useCallback(async (): Promise<MediaStream> => {
    if (streamHasLiveAudio(warmStreamRef.current)) {
      return warmStreamRef.current;
    }
    // A dead warm stream (device unplugged / tracks ended) — drop it and reopen.
    if (warmStreamRef.current) releaseWarmStream();
    await assertMediaDevices();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: AUDIO_CONSTRAINTS,
    });
    if (keepWarm) warmStreamRef.current = stream;
    return stream;
  }, [assertMediaDevices, keepWarm, releaseWarmStream]);

  /**
   * Open the microphone ahead of the first recording so the initial dictation
   * doesn't clip. No-op unless ``keepWarm`` is set. Safe to call repeatedly and
   * safe to fail (e.g. permission not yet granted) — ``startRecording`` still
   * surfaces a real error if capture is genuinely unavailable.
   */
  const prewarm = useCallback(async () => {
    if (!keepWarm) return;
    clearIdleRelease();
    try {
      await acquireStream();
    } catch {
      // Permission missing / device busy — recording will report it properly.
    }
    scheduleIdleRelease();
  }, [keepWarm, clearIdleRelease, acquireStream, scheduleIdleRelease]);

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      chunksRef.current = [];
      cancelledRef.current = false;
      setDuration(0);
      clearIdleRelease();

      // Reuse the warm stream when present (instant); otherwise open one now.
      const stream = await acquireStream();
      streamRef.current = stream;

      // Create MediaRecorder with preferred MIME type
      const options: MediaRecorderOptions = {
        mimeType: 'audio/webm;codecs=opus',
      };

      // Fallback to default if webm not supported
      if (!MediaRecorder.isTypeSupported(options.mimeType!)) {
        delete options.mimeType;
      }

      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        // Snapshot the cancellation flag and recorded duration immediately —
        // cancelRecording() clears chunks and sets cancelledRef synchronously
        // before this async handler runs, so we must check it first.
        const wasCancelled = cancelledRef.current;
        const recordedDuration = startTimeRef.current
          ? (Date.now() - startTimeRef.current) / 1000
          : undefined;

        const webmBlob = new Blob(chunksRef.current, { type: 'audio/webm' });

        // Release the device unless we're keeping it warm for the next capture.
        // When warm, the tracks stay live and an idle timer will reclaim them.
        if (keepWarm) {
          streamRef.current = null;
          scheduleIdleRelease();
        } else {
          streamRef.current?.getTracks().forEach((track) => {
            track.stop();
          });
          streamRef.current = null;
        }

        // Don't fire completion callback if the recording was cancelled
        if (wasCancelled) return;

        // Convert to WAV format to avoid needing ffmpeg on backend
        try {
          const wavBlob = await convertToWav(webmBlob);
          onRecordingComplete?.(wavBlob, recordedDuration);
        } catch (err) {
          console.error('Error converting audio to WAV:', err);
          // Fallback to original blob if conversion fails
          onRecordingComplete?.(webmBlob, recordedDuration);
        }
      };

      mediaRecorder.onerror = (event) => {
        setError('Recording error occurred');
        console.error('MediaRecorder error:', event);
      };

      // WebKit's MediaRecorder drops the WebM EBML header from chunks when
      // started with a timeslice, so concatenated blobs fail to parse in
      // both AudioContext and ffmpeg. Starting with no timeslice produces
      // exactly one dataavailable on stop() with a valid container.
      mediaRecorder.start();
      setIsRecording(true);
      startTimeRef.current = Date.now();

      // Start timer
      timerRef.current = window.setInterval(() => {
        if (startTimeRef.current) {
          const elapsed = (Date.now() - startTimeRef.current) / 1000;
          setDuration(elapsed);

          // Auto-stop at max duration when the caller opts in — dictation
          // sessions pass undefined and run until the user releases the
          // chord or hits stop; voice-clone sample recorders pass 29s to
          // keep reference clips short.
          if (maxDurationSeconds !== undefined && elapsed >= maxDurationSeconds) {
            if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
              mediaRecorderRef.current.stop();
              setIsRecording(false);
              if (timerRef.current !== null) {
                clearInterval(timerRef.current);
                timerRef.current = null;
              }
            }
          }
        }
      }, 100);
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : 'Failed to access microphone. Please check permissions.';
      setError(errorMessage);
      setIsRecording(false);
    }
  }, [
    maxDurationSeconds,
    onRecordingComplete,
    acquireStream,
    keepWarm,
    clearIdleRelease,
    scheduleIdleRelease,
  ]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);

      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  }, [isRecording]);

  const cancelRecording = useCallback(() => {
    if (mediaRecorderRef.current) {
      cancelledRef.current = true; // Must be set before stop() triggers onstop
      chunksRef.current = [];
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setDuration(0);
    }

    // Keep the device warm for the next capture when opted in; otherwise stop
    // the tracks so the mic is released immediately.
    if (keepWarm) {
      streamRef.current = null;
      scheduleIdleRelease();
    } else {
      streamRef.current?.getTracks().forEach((track) => {
        track.stop();
      });
      streamRef.current = null;
    }

    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, [keepWarm, scheduleIdleRelease]);

  // Cleanup on unmount — always fully release the device, warm or not.
  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
      }
      if (idleReleaseTimerRef.current !== null) {
        window.clearTimeout(idleReleaseTimerRef.current);
      }
      streamRef.current?.getTracks().forEach((track) => {
        track.stop();
      });
      warmStreamRef.current?.getTracks().forEach((track) => {
        track.stop();
      });
    };
  }, []);

  return {
    isRecording,
    duration,
    error,
    startRecording,
    stopRecording,
    cancelRecording,
    prewarm,
  };
}
