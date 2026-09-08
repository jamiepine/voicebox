import { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { convertToWav } from '@/lib/utils/audio';
import { usePlatform } from '@/platform/PlatformContext';

/**
 * Why a recording attempt failed, so callers can pick a recovery affordance
 * instead of dumping the browser's own prose at the user. `permission-denied`
 * is the one that needs a way out — macOS never re-prompts once TCC has a
 * denial on record, so the UI has to deep-link the user into System Settings.
 */
export type RecordingErrorKind =
  | 'permission-denied'
  | 'no-device'
  | 'device-busy'
  | 'unavailable'
  | 'unknown';

/**
 * Map a `getUserMedia` rejection onto {@link RecordingErrorKind}. The spec-defined
 * `DOMException.name` is the only stable signal here — `message` is wildly
 * engine-specific (WebKit's denial reads "The request is not allowed by the user
 * agent or the platform in the current context…") and must never reach the UI.
 */
function classifyMediaError(err: unknown): RecordingErrorKind {
  const name = err instanceof Error ? err.name : '';
  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'permission-denied';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'no-device';
    case 'NotReadableError':
    case 'AbortError':
      return 'device-busy';
    default:
      return 'unknown';
  }
}

/**
 * `navigator.mediaDevices` itself is missing — a stale WKWebView, a non-secure
 * origin on the web build. Distinct from a denial: there is no permission for
 * the user to flip, so the recovery advice differs.
 */
class MediaUnavailableError extends Error {
  constructor() {
    super('mediaDevices unavailable');
    this.name = 'MediaUnavailableError';
  }
}

interface UseAudioRecordingOptions {
  maxDurationSeconds?: number;
  onRecordingComplete?: (blob: Blob, duration?: number) => void;
}

/**
 * Record a single audio clip from the user's microphone.
 *
 * Wraps `getUserMedia` plus `MediaRecorder` and hands the caller a WAV blob
 * through `onRecordingComplete`, falling back to the raw WebM when conversion
 * fails. Pass `maxDurationSeconds` to auto-stop (voice-clone samples use 29s);
 * omit it for open-ended dictation that runs until `stopRecording`.
 *
 * Failures surface as a translated `error` string plus an `errorKind` the UI
 * can branch on, so a denied microphone can offer a way to grant it rather
 * than printing the browser's own wording.
 */
export function useAudioRecording({
  maxDurationSeconds,
  onRecordingComplete,
}: UseAudioRecordingOptions = {}) {
  const platform = usePlatform();
  const { t } = useTranslation();
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [errorKind, setErrorKind] = useState<RecordingErrorKind | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const startTimeRef = useRef<number | null>(null);
  const cancelledRef = useRef<boolean>(false);

  /**
   * Translate a {@link RecordingErrorKind} into the sentence shown to the
   * user. `unavailable` splits by platform: on desktop the microphone is a
   * system grant, in the browser it is usually a non-secure origin.
   */
  const describeError = useCallback(
    (kind: RecordingErrorKind): string => {
      switch (kind) {
        case 'permission-denied':
          return t('audioSample.errors.permissionDenied');
        case 'no-device':
          return t('audioSample.errors.noDevice');
        case 'device-busy':
          return t('audioSample.errors.deviceBusy');
        case 'unavailable':
          return platform.metadata.isTauri
            ? t('audioSample.errors.unavailableTauri')
            : t('audioSample.errors.unavailableWeb');
        default:
          return t('audioSample.errors.unknown');
      }
    },
    [t, platform.metadata.isTauri],
  );

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      setErrorKind(null);
      chunksRef.current = [];
      cancelledRef.current = false;
      setDuration(0);

      // Check if getUserMedia is available
      // In Tauri, navigator.mediaDevices might not be available immediately
      if (typeof navigator === 'undefined') {
        throw new MediaUnavailableError();
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        // Try waiting a bit for Tauri webview to initialize
        await new Promise((resolve) => setTimeout(resolve, 100));

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          console.error('MediaDevices check:', {
            hasNavigator: typeof navigator !== 'undefined',
            hasMediaDevices: !!navigator?.mediaDevices,
            hasGetUserMedia: !!navigator?.mediaDevices?.getUserMedia,
            isTauri: platform.metadata.isTauri,
          });

          throw new MediaUnavailableError();
        }
      }

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

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

        // Stop all tracks now that we have the data
        streamRef.current?.getTracks().forEach((track) => {
          track.stop();
        });
        streamRef.current = null;

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
        setError(t('audioSample.errors.recorderFailed'));
        setErrorKind('unknown');
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
      const kind = err instanceof MediaUnavailableError ? 'unavailable' : classifyMediaError(err);
      // Keep the raw rejection in the console for bug reports — only the
      // translated summary reaches the UI.
      console.error('Failed to start recording:', err);
      setErrorKind(kind);
      setError(describeError(kind));
      setIsRecording(false);
    }
  }, [maxDurationSeconds, onRecordingComplete, platform.metadata.isTauri, t, describeError]);

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

    // Stop all tracks
    streamRef.current?.getTracks().forEach((track) => {
      track.stop();
    });
    streamRef.current = null;

    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
      }
      streamRef.current?.getTracks().forEach((track) => {
        track.stop();
      });
    };
  }, []);

  return {
    isRecording,
    duration,
    error,
    errorKind,
    startRecording,
    stopRecording,
    cancelRecording,
  };
}
