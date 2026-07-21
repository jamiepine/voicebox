import { useCallback, useEffect, useRef, useState } from 'react';
import { usePlatform } from '@/platform/PlatformContext';
import { convertToWav } from '@/lib/utils/audio';

interface UseAudioRecordingOptions {
  maxDurationSeconds?: number;
  // ``context`` is whatever was handed to ``startRecording`` for this take,
  // threaded back untouched so callers can correlate the result with the
  // recording it came from (the dictate window pairs it with the focus
  // snapshot captured at chord-start).
  onRecordingComplete?: (blob: Blob, duration?: number, context?: unknown) => void;
  /**
   * Keep the microphone ``MediaStream`` open between recordings instead of
   * tearing it down on every stop. This is what removes the "first words get
   * clipped" problem on push-to-talk dictation: ``getUserMedia`` on macOS can
   * take several hundred ms — up to a second cold — to hand back a stream, and
   * ``MediaRecorder`` only starts capturing *after* it resolves, so everything
   * spoken in that window is lost. With a warm stream already open, the next
   * ``startRecording`` skips ``getUserMedia`` entirely.
   *
   * Off by default: the voice-clone sample recorders release the device
   * immediately, and the dictation session only opts in when the user enables
   * the "keep microphone ready" setting. While on, the warm stream stays open —
   * and the OS mic-in-use indicator stays lit — until it's explicitly released
   * (dictation disabled or the setting turned off), so the trade-off is visible
   * and user-controlled rather than a background mic that's always warm.
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
  const timerRef = useRef<number | null>(null);
  const startTimeRef = useRef<number | null>(null);
  const cancelledRef = useRef<boolean>(false);
  // Mirror of ``isRecording`` for reads inside callbacks that would otherwise
  // close over a stale render.
  const isRecordingRef = useRef(false);
  // A ``getUserMedia`` call in flight, shared so concurrent acquirers (prewarm
  // plus an immediate chord) coalesce onto one stream instead of each opening —
  // and orphaning — their own.
  const acquiringRef = useRef<Promise<MediaStream> | null>(null);
  // True from ``startRecording`` entry until the recorder is actually running
  // (or has failed), so a stop that arrives mid-acquisition can be deferred.
  const startingRef = useRef(false);
  // True from MediaRecorder.stop() until onstop has snapshotted the take's
  // shared refs. React state and MediaRecorder.state both flip before onstop,
  // so without this gate a rapid next chord can clear chunks/duration/cancel
  // state out from under the recorder that is still finalising.
  const finishingRef = useRef(false);
  const pendingStopRef = useRef(false);
  // Bumped per recording so a stale recorder's ``onstop`` can tell it's no
  // longer the active one before it touches the shared stream refs.
  const recordingCounterRef = useRef(0);
  // Bumped whenever the warm stream is released/aborted so a ``getUserMedia``
  // still in flight can tell its result is stale and stop it instead of
  // adopting a live mic after disable/unmount.
  const acquireGenRef = useRef(0);
  // Set when a release is requested mid-recording; the onstop path performs the
  // deferred release once capture finishes rather than yanking the device now.
  const releaseAfterStopRef = useRef(false);

  // Keeps the ref in lockstep with the state so the synchronous stop path reads
  // a fresh value without waiting for a rerender.
  const setRecording = useCallback((next: boolean) => {
    isRecordingRef.current = next;
    setIsRecording(next);
  }, []);

  const releaseWarmStream = useCallback(() => {
    // Invalidate any getUserMedia still in flight so its stream is stopped on
    // resolve rather than adopted as the warm stream.
    acquireGenRef.current += 1;
    // Don't tear the device out from under an active/starting recording — the
    // warm stream is the one backing it; defer to the onstop path instead.
    if (isRecordingRef.current || startingRef.current) {
      releaseAfterStopRef.current = true;
      return;
    }
    warmStreamRef.current?.getTracks().forEach((track) => {
      track.stop();
    });
    warmStreamRef.current = null;
  }, []);

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
    // Captured separately so it stays typed as the full stream after the live
    // check narrows ``warmStreamRef.current`` itself.
    const existing = warmStreamRef.current;
    if (streamHasLiveAudio(warmStreamRef.current)) {
      return warmStreamRef.current;
    }
    // Coalesce concurrent acquirers onto one getUserMedia call so prewarm and
    // an immediate chord can't open two streams.
    if (acquiringRef.current) return acquiringRef.current;
    // A dead warm stream (device unplugged / tracks ended) — drop it and reopen.
    if (existing) {
      existing.getTracks().forEach((track) => {
        track.stop();
      });
      warmStreamRef.current = null;
    }
    const gen = acquireGenRef.current;
    const acquisition = (async () => {
      await assertMediaDevices();
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: AUDIO_CONSTRAINTS,
      });
      // Released / disabled / unmounted while acquiring — this stream is stale,
      // so stop it instead of leaving a live mic open, and abort the caller.
      if (gen !== acquireGenRef.current) {
        stream.getTracks().forEach((track) => {
          track.stop();
        });
        throw new Error('microphone acquisition aborted');
      }
      if (keepWarm) warmStreamRef.current = stream;
      return stream;
    })();
    acquiringRef.current = acquisition;
    try {
      return await acquisition;
    } finally {
      if (acquiringRef.current === acquisition) acquiringRef.current = null;
    }
  }, [assertMediaDevices, keepWarm]);

  /**
   * Open the microphone ahead of the first recording so the initial dictation
   * doesn't clip. No-op unless ``keepWarm`` is set. Safe to call repeatedly and
   * safe to fail (e.g. permission not yet granted) — ``startRecording`` still
   * surfaces a real error if capture is genuinely unavailable.
   */
  const prewarm = useCallback(async () => {
    if (!keepWarm) return;
    try {
      await acquireStream();
    } catch {
      // Permission missing / device busy / aborted — recording will report a
      // real error if capture is genuinely unavailable.
    }
  }, [keepWarm, acquireStream]);

  const startRecording = useCallback(
    async (context?: unknown) => {
      // A second chord can arrive while the first one is still waiting on
      // getUserMedia. Never create overlapping MediaRecorders on the same
      // coalesced stream; the original take will honor any deferred stop.
      if (
        startingRef.current ||
        finishingRef.current ||
        mediaRecorderRef.current?.state === 'recording'
      )
        return;
      startingRef.current = true;
      pendingStopRef.current = false;
      // A new recording supersedes any release deferred from a prior take.
      releaseAfterStopRef.current = false;
      const recordingId = ++recordingCounterRef.current;
      try {
        setError(null);
        chunksRef.current = [];
        cancelledRef.current = false;
        setDuration(0);

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
          // Whether this recorder is still the active one. A stale onstop (an
          // older recorder stopping after a newer startRecording) must not touch
          // the shared stream refs.
          const isCurrent = recordingCounterRef.current === recordingId;
          // Snapshot the cancellation flag and recorded duration immediately —
          // cancelRecording() clears chunks and sets cancelledRef synchronously
          // before this async handler runs, so we must check it first.
          const wasCancelled = cancelledRef.current;
          const recordedDuration = startTimeRef.current
            ? (Date.now() - startTimeRef.current) / 1000
            : undefined;

          const webmBlob = new Blob(chunksRef.current, { type: 'audio/webm' });

          // Release the device unless we're keeping it warm for the next capture.
          // Act on this recorder's own stream; only touch the shared refs when
          // this is still the current recording.
          if (keepWarm) {
            if (isCurrent) {
              streamRef.current = null;
              // A release requested mid-recording (dictation disabled) is
              // honored now that capture has finished; otherwise the warm
              // stream stays open for the next take.
              if (releaseAfterStopRef.current) {
                releaseAfterStopRef.current = false;
                releaseWarmStream();
              }
            }
          } else {
            stream.getTracks().forEach((track) => {
              track.stop();
            });
            if (isCurrent) streamRef.current = null;
          }

          // All shared per-take refs have now been snapshotted and stream
          // cleanup is complete. A new take may begin while WAV conversion and
          // upload continue using the local values above.
          finishingRef.current = false;

          // Don't fire completion callback if the recording was cancelled
          if (wasCancelled) return;

          // Convert to WAV format to avoid needing ffmpeg on backend
          try {
            const wavBlob = await convertToWav(webmBlob);
            onRecordingComplete?.(wavBlob, recordedDuration, context);
          } catch (err) {
            console.error('Error converting audio to WAV:', err);
            // Fallback to original blob if conversion fails
            onRecordingComplete?.(webmBlob, recordedDuration, context);
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
        setRecording(true);
        startTimeRef.current = Date.now();
        startingRef.current = false;

        // A stop (chord release) that landed while the mic was still opening —
        // honor it now that capture has actually begun.
        if (pendingStopRef.current) {
          pendingStopRef.current = false;
          finishingRef.current = true;
          mediaRecorder.stop();
          setRecording(false);
          return;
        }

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
                finishingRef.current = true;
                mediaRecorderRef.current.stop();
                setRecording(false);
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
        // A fresh (non-warm) stream opened before the failure must be released
        // so the mic doesn't stay lit; a warm stream is reusable, so it's kept.
        if (!keepWarm) {
          streamRef.current?.getTracks().forEach((track) => {
            track.stop();
          });
          streamRef.current = null;
        }
        startingRef.current = false;
        finishingRef.current = false;
        pendingStopRef.current = false;
        setError(errorMessage);
        setRecording(false);
      }
    },
    [
      maxDurationSeconds,
      onRecordingComplete,
      acquireStream,
      keepWarm,
      releaseWarmStream,
      setRecording,
    ],
  );

  const stopRecording = useCallback(() => {
    // The recorder's own state is the lifecycle authority — React ``isRecording``
    // lags a render behind ``mediaRecorder.start()``, so a chord release in that
    // window would otherwise be dropped.
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state === 'recording') {
      finishingRef.current = true;
      recorder.stop();
      setRecording(false);

      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    } else if (startingRef.current) {
      // Stop arrived before capture began (mic still opening) — defer it so
      // startRecording stops as soon as the recorder goes live.
      pendingStopRef.current = true;
    }
  }, [setRecording]);

  const cancelRecording = useCallback(() => {
    cancelledRef.current = true; // Must be set before stop() triggers onstop
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      chunksRef.current = [];
      finishingRef.current = true;
      recorder.stop();
      setRecording(false);
      setDuration(0);
    } else if (startingRef.current) {
      // Cancel during mic acquisition — stop as soon as capture begins; the
      // cancelled flag suppresses the completion callback.
      pendingStopRef.current = true;
    }

    // Keep the device warm for the next capture when opted in; otherwise stop
    // the tracks so the mic is released immediately.
    if (keepWarm) {
      streamRef.current = null;
      if (releaseAfterStopRef.current) {
        releaseAfterStopRef.current = false;
        releaseWarmStream();
      }
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
  }, [keepWarm, releaseWarmStream, setRecording]);

  // Cleanup on unmount — always fully release the device, warm or not.
  useEffect(() => {
    return () => {
      // Invalidate any in-flight acquisition so a stream resolving after unmount
      // stops itself instead of leaking a live mic.
      acquireGenRef.current += 1;
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
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
    releaseWarm: releaseWarmStream,
  };
}
