import { useMutation, useQueryClient } from '@tanstack/react-query';
import { emit as tauriEmit } from '@tauri-apps/api/event';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { PillState } from '@/components/CapturePill/CapturePill';
import { apiClient } from '@/lib/api/client';
import type {
  CaptureListResponse,
  CaptureResponse,
  CaptureSource,
} from '@/lib/api/types';
import { useAudioRecording } from '@/lib/hooks/useAudioRecording';

/**
 * Broadcast to sibling Tauri webviews that the captures list has changed.
 * The main CapturesTab listens, seeds its React Query cache, and focuses the
 * new row, so uploads from the floating dictate window show up live.
 *
 * ``capture:created`` carries the full response so the sibling can seed its
 * cache before the refetch lands — otherwise the selection-guard effect
 * would snap back to ``captures[0]`` in the race window between
 * ``setSelectedId(new)`` and the list actually containing the new row.
 *
 * No-op in web mode — there are no siblings to notify.
 */
function broadcastCreated(capture: CaptureResponse) {
  tauriEmit('capture:created', { capture }).catch(() => {
    /* not running inside Tauri; nothing to sync to */
  });
}

function broadcastUpdated(id: string) {
  tauriEmit('capture:updated', { id }).catch(() => {
    /* not running inside Tauri; nothing to sync to */
  });
}

const REST_FADE_MS = 900;
// How long the green "Done" pill stays visible after refine (or transcribe,
// when auto-refine is off) completes, before the fade-out begins.
const COMPLETED_DWELL_MS = 2000;
// Long enough to read a full backend stack message and click-to-copy.
const ERROR_PILL_VISIBLE_MS = 6000;
// Short self-explanatory notices (e.g. "Recording too short, canceled") —
// there's nothing to read or copy, so clear out quickly.
const BRIEF_NOTICE_MS = 2000;
// MediaRecorder.start(100) emits its first chunk ~100ms in, but the webm
// container header isn't guaranteed to be finalised that quickly — anything
// under half a second tends to produce a blob neither AudioContext.decode
// nor ffmpeg will accept. Caught client-side and surfaced as a friendly
// "Recording too short, canceled" pill instead of bubbling up a 400.
const MIN_RECORDING_DURATION_S = 0.5;
const SHORT_RECORDING_MESSAGE = 'Recording too short, canceled';

export type CapturePillState = PillState | 'hidden';

export interface UseCaptureRecordingSessionOptions {
  /**
   * Fired after a capture row is created on the server. Callers can use this
   * to select the new capture or emit a Tauri event to a sibling window.
   */
  onCaptureCreated?: (capture: CaptureResponse) => void;
  /**
   * Fired with the final delivered text — refined if ``auto_refine`` was on
   * for this capture, raw transcript otherwise. Used by the floating
   * dictate window to hand the text off to the Rust auto-paste pipeline.
   *
   * ``allowAutoPaste`` snapshots the setting at chord-start so a refine that
   * lands after the user flips the toggle still uses the value the capture
   * was created under.
   */
  onFinalText?: (
    text: string,
    capture: CaptureResponse,
    allowAutoPaste: boolean,
  ) => void;
}

export interface UseCaptureRecordingSessionResult {
  pillState: CapturePillState;
  pillElapsedMs: number;
  errorMessage: string | null;
  isRecording: boolean;
  isUploading: boolean;
  isRefining: boolean;
  startRecording: () => void;
  stopRecording: () => void;
  toggleRecording: () => void;
  dismissError: () => void;
  uploadFile: (file: File, source: CaptureSource) => void;
  refine: (captureId: string) => void;
}

/**
 * Owns the full record → transcribe → refine → rest lifecycle behind the
 * capture pill. The pill component and the Dictate/Stop button are the only
 * consumers; everything else (cache seeding, error toasts, settings reads) is
 * internal so the hook can be reused from a floating Tauri window without the
 * containing tab.
 */
export function useCaptureRecordingSession(
  options: UseCaptureRecordingSessionOptions = {},
): UseCaptureRecordingSessionResult {
  const queryClient = useQueryClient();
  // Every capture setting is resolved server-side. ``stt_model``,
  // ``llm_model`` and refine flags are read from the capture_settings table
  // inside POST /captures and /captures/*/refine, and ``auto_refine`` comes
  // back on the create response so the client decides whether to chain a
  // refine call using a value that can't go stale across sibling webviews.

  const [pillState, setPillState] = useState<CapturePillState>('hidden');
  const [frozenElapsedMs, setFrozenElapsedMs] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const restTimerRef = useRef<number | null>(null);
  const errorTimerRef = useRef<number | null>(null);

  // Mutation callbacks close over stale pillState otherwise.
  const pillStateRef = useRef<CapturePillState>('hidden');
  pillStateRef.current = pillState;

  const onCaptureCreatedRef = useRef(options.onCaptureCreated);
  onCaptureCreatedRef.current = options.onCaptureCreated;

  const onFinalTextRef = useRef(options.onFinalText);
  onFinalTextRef.current = options.onFinalText;

  // Snapshot of ``allow_auto_paste`` from the capture-create response —
  // held so the refine onSuccess (which only sees the plain CaptureResponse)
  // can still pass the original setting through to onFinalText.
  const allowAutoPasteRef = useRef<boolean>(true);

  const clearRestTimer = useCallback(() => {
    if (restTimerRef.current !== null) {
      window.clearTimeout(restTimerRef.current);
      restTimerRef.current = null;
    }
  }, []);

  const clearErrorTimer = useCallback(() => {
    if (errorTimerRef.current !== null) {
      window.clearTimeout(errorTimerRef.current);
      errorTimerRef.current = null;
    }
  }, []);

  const scheduleHidePill = useCallback(() => {
    clearRestTimer();
    setPillState('completed');
    // Two-hop timer: show the green "Done" pill for COMPLETED_DWELL_MS,
    // then hand off to the existing rest-fade before unmounting.
    restTimerRef.current = window.setTimeout(() => {
      setPillState('rest');
      restTimerRef.current = window.setTimeout(() => {
        setPillState('hidden');
        restTimerRef.current = null;
      }, REST_FADE_MS);
    }, COMPLETED_DWELL_MS);
  }, [clearRestTimer]);

  const showError = useCallback(
    (message: string, durationMs: number = ERROR_PILL_VISIBLE_MS) => {
      clearRestTimer();
      clearErrorTimer();
      setErrorMessage(message || 'Something went wrong');
      setPillState('error');
      errorTimerRef.current = window.setTimeout(() => {
        setPillState('hidden');
        setErrorMessage(null);
        errorTimerRef.current = null;
      }, durationMs);
    },
    [clearRestTimer, clearErrorTimer],
  );

  const dismissError = useCallback(() => {
    clearErrorTimer();
    setPillState('hidden');
    setErrorMessage(null);
  }, [clearErrorTimer]);

  useEffect(
    () => () => {
      clearRestTimer();
      clearErrorTimer();
    },
    [clearRestTimer, clearErrorTimer],
  );

  const refineMutation = useMutation({
    // Empty body — backend resolves flags and model from capture_settings.
    mutationFn: async (captureId: string) => apiClient.refineCapture(captureId, {}),
    onSuccess: (data, captureId) => {
      queryClient.invalidateQueries({ queryKey: ['captures'] });
      broadcastUpdated(captureId);
      if (pillStateRef.current === 'refining') scheduleHidePill();
      const finalText = data.transcript_refined ?? data.transcript_raw;
      if (finalText) {
        onFinalTextRef.current?.(finalText, data, allowAutoPasteRef.current);
      }
    },
    onError: (err: Error) => {
      showError(err.message || 'Refinement failed');
    },
  });

  const uploadMutation = useMutation({
    mutationFn: async ({ file, source }: { file: File; source: CaptureSource }) =>
      apiClient.createCapture(file, { source }),
    onSuccess: (capture) => {
      queryClient.setQueryData<CaptureListResponse>(['captures'], (prev) => {
        if (!prev) return prev;
        if (prev.items.some((c) => c.id === capture.id)) return prev;
        return { ...prev, items: [capture, ...prev.items], total: prev.total + 1 };
      });
      queryClient.invalidateQueries({ queryKey: ['captures'] });
      broadcastCreated(capture);
      onCaptureCreatedRef.current?.(capture);
      allowAutoPasteRef.current = capture.allow_auto_paste;
      if (capture.auto_refine) {
        setPillState('refining');
        refineMutation.mutate(capture.id);
      } else {
        if (pillStateRef.current === 'transcribing') scheduleHidePill();
        if (capture.transcript_raw) {
          onFinalTextRef.current?.(
            capture.transcript_raw,
            capture,
            capture.allow_auto_paste,
          );
        }
      }
    },
    onError: (err: Error) => {
      // Backend's librosa-audioread fallback returns a 400 with this shape
      // for tiny/corrupt webm blobs that slip past the client guard —
      // translate it to the same friendly message so the user sees one
      // consistent cause, not an opaque decode error.
      const msg = err.message || '';
      if (/could not decode/i.test(msg) || /empty or corrupt/i.test(msg)) {
        showError(SHORT_RECORDING_MESSAGE, BRIEF_NOTICE_MS);
      } else {
        showError(msg || 'Upload failed');
      }
    },
  });

  const {
    isRecording,
    duration,
    startRecording: beginAudioRecording,
    stopRecording,
    error: recordError,
  } = useAudioRecording({
    onRecordingComplete: (blob, recordedDuration) => {
      // Trigger-happy tap — MediaRecorder hasn't emitted a usable chunk yet
      // so the blob is empty or unparseable. Surface it as a transient pill
      // so the user sees their recording was recognised and canceled.
      if (!blob.size || (recordedDuration ?? 0) < MIN_RECORDING_DURATION_S) {
        showError(SHORT_RECORDING_MESSAGE, BRIEF_NOTICE_MS);
        return;
      }
      setFrozenElapsedMs(Math.round((recordedDuration ?? 0) * 1000));
      setPillState('transcribing');
      const extension = blob.type.includes('wav')
        ? 'wav'
        : blob.type.includes('webm')
          ? 'webm'
          : 'bin';
      const file = new File([blob], `dictation-${Date.now()}.${extension}`, {
        type: blob.type,
      });
      uploadMutation.mutate({ file, source: 'dictation' });
    },
  });

  useEffect(() => {
    if (recordError) {
      showError(recordError);
    }
  }, [recordError, showError]);

  const startRecording = useCallback(() => {
    if (isRecording) return;
    clearRestTimer();
    setFrozenElapsedMs(0);
    setPillState('recording');
    beginAudioRecording();
  }, [isRecording, beginAudioRecording, clearRestTimer]);

  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
      return;
    }
    startRecording();
  }, [isRecording, startRecording, stopRecording]);

  const uploadFile = useCallback(
    (file: File, source: CaptureSource) => {
      uploadMutation.mutate({ file, source });
    },
    [uploadMutation],
  );

  const refine = useCallback(
    (captureId: string) => {
      refineMutation.mutate(captureId);
    },
    [refineMutation],
  );

  const pillElapsedMs =
    pillState === 'recording' ? Math.round(duration * 1000) : frozenElapsedMs;

  return {
    pillState,
    pillElapsedMs,
    errorMessage,
    isRecording,
    isUploading: uploadMutation.isPending,
    isRefining: refineMutation.isPending,
    startRecording,
    stopRecording,
    toggleRecording,
    dismissError,
    uploadFile,
    refine,
  };
}
