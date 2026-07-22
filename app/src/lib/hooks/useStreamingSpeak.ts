/**
 * React hook for the progressive-playback path — talks to
 * ``POST /speak/stream`` and schedules each sentence's audio against a
 * shared ``AudioContext`` so the browser starts speaking before the LLM
 * has finished generating.
 *
 * The hook owns:
 *   - state machine (idle → connecting → streaming → complete | error)
 *   - single ``AudioContext`` created lazily on first ``speak()``
 *   - a ``nextStartTime`` cursor that keeps sentence-N+1 abutting
 *     sentence-N so there's no gap between chunks even when audio
 *     arrives faster than real time
 *   - the ``AbortController`` for the fetch, wired to a ``.abort()``
 *     method so the UI can cancel a runaway generation
 *   - a subtitle log accumulated from the ``text`` field on each audio
 *     frame, ready for a captions overlay
 *
 * The audio payload is base64 PCM float32 mono, decoded with
 * ``atob`` → ``Uint8Array`` → ``Float32Array`` → ``AudioBuffer``. No
 * dependency on external decoders.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { apiClient } from '@/lib/api/client';
import type {
  SpeakStreamEvent,
  StreamingSpeakRequest,
} from '@/lib/api/types';

export type StreamingSpeakStatus =
  | 'idle'
  | 'connecting'
  | 'streaming'
  | 'complete'
  | 'error'
  | 'aborted';

export interface StreamingSpeakSentence {
  index: number;
  text: string;
  /** Wall-clock ``AudioContext.currentTime`` value the chunk is scheduled to start at. */
  scheduledAt: number;
  /** Duration in seconds of this chunk's audio. */
  duration: number;
}

export interface StreamingSpeakState {
  status: StreamingSpeakStatus;
  generationId: string | null;
  streamingLlm: boolean;
  sentences: StreamingSpeakSentence[];
  sampleRate: number | null;
  duration: number | null;
  audioPath: string | null;
  error: string | null;
  /** Best-effort estimate of which sentence is currently playing, or -1 before playback starts. */
  playingIndex: number;
}

const INITIAL_STATE: StreamingSpeakState = {
  status: 'idle',
  generationId: null,
  streamingLlm: false,
  sentences: [],
  sampleRate: null,
  duration: null,
  audioPath: null,
  error: null,
  playingIndex: -1,
};

export function useStreamingSpeak() {
  const [state, setState] = useState<StreamingSpeakState>(INITIAL_STATE);

  // Refs so the async fetch loop reads the latest handles without going
  // through React state updates (which would trigger re-renders per
  // audio chunk).
  const audioCtxRef = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const abortRef = useRef<AbortController | null>(null);
  const activeSourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const playingIndexRef = useRef<number>(-1);

  const _teardown = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    for (const src of activeSourcesRef.current) {
      try {
        src.stop();
      } catch {
        // already ended
      }
      src.disconnect();
    }
    activeSourcesRef.current = [];
    if (audioCtxRef.current) {
      // ``AudioContext.close`` is async but we don't await it — the ref
      // is nulled synchronously so the next speak() opens a fresh one.
      audioCtxRef.current.close().catch(() => {
        /* ignore — already closed */
      });
      audioCtxRef.current = null;
    }
    nextStartTimeRef.current = 0;
    playingIndexRef.current = -1;
  }, []);

  // Clean up on unmount so a mid-stream navigation doesn't leave an
  // orphan AudioContext playing.
  useEffect(() => {
    return () => {
      _teardown();
    };
  }, [_teardown]);

  const abort = useCallback(() => {
    _teardown();
    setState((prev) =>
      prev.status === 'idle' || prev.status === 'complete' || prev.status === 'error'
        ? prev
        : { ...prev, status: 'aborted' },
    );
  }, [_teardown]);

  const speak = useCallback(
    async (params: StreamingSpeakRequest) => {
      // Any in-flight stream gets cleaned up before we start the next one
      // — the hook is single-track by design.
      _teardown();
      setState({ ...INITIAL_STATE, status: 'connecting' });

      // AudioContext must be created inside a user gesture to satisfy
      // Chrome's autoplay policy — callers wire speak() to a click.
      // Some browsers still start suspended; resume() ensures playback.
      const ctx = new AudioContext();
      audioCtxRef.current = ctx;
      try {
        await ctx.resume();
      } catch {
        // resume rejects on already-running contexts on some engines
      }
      nextStartTimeRef.current = ctx.currentTime;

      const controller = new AbortController();
      abortRef.current = controller;

      const handleEvent = (event: SpeakStreamEvent) => {
        if (event.type === 'meta') {
          setState((prev) => ({
            ...prev,
            status: 'streaming',
            generationId: event.generation_id,
            streamingLlm: event.streaming_llm,
            sampleRate: event.sample_rate,
          }));
          return;
        }
        if (event.type === 'audio') {
          const audioCtx = audioCtxRef.current;
          if (!audioCtx) return;

          const pcm = _base64ToFloat32(event.pcm_base64);
          if (pcm.length === 0) return;

          // ``sample_rate`` was pinned on the meta frame; fall back to
          // the AudioContext's rate if a client somehow skipped meta.
          const sampleRate =
            audioCtx.sampleRate === 0 ? 24000 : audioCtx.sampleRate;
          const buffer = audioCtx.createBuffer(1, pcm.length, sampleRateOrDefault(audioCtx, 24000));
          buffer.copyToChannel(pcm, 0);

          const source = audioCtx.createBufferSource();
          source.buffer = buffer;
          source.connect(audioCtx.destination);

          const startAt = Math.max(audioCtx.currentTime, nextStartTimeRef.current);
          const chunkDuration = pcm.length / (buffer.sampleRate || sampleRate);

          const scheduledIndex = event.sentence_index;
          source.onended = () => {
            source.disconnect();
            activeSourcesRef.current = activeSourcesRef.current.filter(
              (s) => s !== source,
            );
            // The current-playing index only moves forward — a chunk that
            // finished after a later one already started shouldn't drag
            // the UI back.
            if (scheduledIndex >= playingIndexRef.current) {
              playingIndexRef.current = scheduledIndex + 1;
              setState((prev) =>
                prev.playingIndex >= scheduledIndex + 1
                  ? prev
                  : { ...prev, playingIndex: scheduledIndex + 1 },
              );
            }
          };
          source.start(startAt);
          activeSourcesRef.current.push(source);

          const sentence: StreamingSpeakSentence = {
            index: event.sentence_index,
            text: event.text,
            scheduledAt: startAt,
            duration: chunkDuration,
          };

          nextStartTimeRef.current = startAt + chunkDuration;

          setState((prev) => ({
            ...prev,
            sentences: [...prev.sentences, sentence],
            playingIndex:
              prev.playingIndex === -1 ? event.sentence_index : prev.playingIndex,
          }));
          if (playingIndexRef.current === -1) {
            playingIndexRef.current = event.sentence_index;
          }
          return;
        }
        if (event.type === 'complete') {
          setState((prev) => ({
            ...prev,
            status: 'complete',
            generationId: event.generation_id,
            duration: event.duration,
            audioPath: event.audio_path ?? null,
          }));
          return;
        }
        if (event.type === 'error') {
          setState((prev) => ({
            ...prev,
            status: 'error',
            generationId: event.generation_id ?? prev.generationId,
            error: event.message,
          }));
          return;
        }
      };

      try {
        await apiClient.streamSpeak(params, handleEvent, controller.signal);
        // ``streamSpeak`` returns on ``[DONE]``. If the server never
        // sent ``complete`` before ``[DONE]`` (e.g. empty output), leave
        // the state where the last event handler put it — either
        // ``error`` or ``streaming``.
        setState((prev) =>
          prev.status === 'streaming' ? { ...prev, status: 'complete' } : prev,
        );
      } catch (err) {
        // Abort races are expected — surface them as ``aborted`` not
        // ``error`` so the UI can distinguish a user cancellation from a
        // real fault.
        if ((err as Error)?.name === 'AbortError') {
          setState((prev) => ({ ...prev, status: 'aborted' }));
          return;
        }
        setState((prev) => ({
          ...prev,
          status: 'error',
          error: (err as Error)?.message ?? 'Streaming speak failed',
        }));
      } finally {
        abortRef.current = null;
      }
    },
    [_teardown],
  );

  return { state, speak, abort };
}

function _base64ToFloat32(b64: string): Float32Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  // The backend serialises native-endian float32 with ``ndarray.tobytes()``.
  // Every platform we ship on (macOS Apple Silicon / x86_64 / Linux x86_64)
  // is little-endian, which matches Web Audio's Float32Array. If we ever
  // support big-endian servers this needs an explicit DataView pass.
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

function sampleRateOrDefault(ctx: AudioContext, fallback: number): number {
  return ctx.sampleRate > 0 ? ctx.sampleRate : fallback;
}
