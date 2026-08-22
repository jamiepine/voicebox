// Single, app-wide EventSource that fans out generation status updates
// from the backend's /events/generations SSE stream.
//
// Background:
//   The original useGenerationProgress hook opened one EventSource per
//   pending generation. Once a user had 6+ generations running, the
//   browser hit the HTTP/1.1 per-origin 6-connection cap and any new
//   POST /generate call was queued behind those EventSources. The user
//   saw the "Send" button as unresponsive.
//
// Fix:
//   - Backend now broadcasts every generation's status changes via
//     /events/generations (a single multiplexed SSE stream).
//   - This module opens exactly one EventSource for that endpoint and
//     re-emits each message as a callback to subscribed handlers.
//   - The EventSource is opened lazily on first subscriber and closed
//     when the last subscriber unsubscribes.
//   - Reconnects use exponential backoff (1s -> 30s) that resets only on
//     a successful "ready" event or when the last subscriber leaves,
//     not on every disconnect, so a downed backend doesn't get hammered.

import { useServerStore } from '@/stores/serverStore';

export interface GenerationStatusEvent {
  id: string;
  status: 'loading_model' | 'generating' | 'completed' | 'failed' | 'not_found';
  duration?: number;
  error?: string;
  source?: string;
}

export type GenerationSubscriber = (data: GenerationStatusEvent) => void;

class GenerationEventBus {
  private source: EventSource | null = null;
  private url: string | null = null;
  private subscribers: Set<GenerationSubscriber> = new Set();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = 1000;
  private static readonly MAX_BACKOFF_MS = 30_000;

  /**
   * Subscribe to generation status updates. Returns an unsubscribe
   * function. The first subscriber opens the EventSource; the last
   * unsubscriber closes it.
   */
  subscribe(fn: GenerationSubscriber): () => void {
    this.subscribers.add(fn);
    this.ensureConnection();
    return () => {
      this.subscribers.delete(fn);
      if (this.subscribers.size === 0) {
        // Last subscriber left: close the socket and reset backoff so
        // the next session starts fresh at 1s, not 30s.
        this.reconnectDelay = 1000;
        this.disconnect();
      }
    };
  }

  /** How many subscribers currently registered. Useful for debugging. */
  subscriberCount(): number {
    return this.subscribers.size;
  }

  /**
   * Override the SSE endpoint URL. Tests / dev overrides use this; in
   * production defaultGenerationEventsUrl() picks the right base URL
   * from useServerStore (dev 127.0.0.1 vs production Tauri sidecar).
   */
  setUrl(url: string): void {
    if (url === this.url) return;
    this.url = url;
    if (this.subscribers.size > 0) {
      this.disconnect();
      this.ensureConnection();
    }
  }

  private ensureConnection(): void {
    if (this.source || typeof window === 'undefined') return;
    const url = this.url ?? defaultGenerationEventsUrl();
    try {
      const source = new EventSource(url);
      // Capture the source at the time the listener is attached so a
      // late-firing onerror from a stale connection (already replaced
      // by a new source after a reconnect) cannot tear down the live
      // connection or schedule a redundant retry.
      const sourceRef = source;
      // A successful "ready" event from the backend means we are talking
      // to it again; reset the backoff so the next outage starts at 1s
      // rather than continuing from whatever value the last outage left.
      source.addEventListener('ready', () => {
        this.reconnectDelay = 1000;
      });
      source.addEventListener('generation', (ev) => {
        try {
          const data = JSON.parse((ev as MessageEvent).data) as GenerationStatusEvent;
          for (const fn of this.subscribers) {
            try {
              fn(data);
            } catch (err) {
              // Don't let one bad subscriber kill the others.
              console.error('generation event subscriber threw', err);
            }
          }
        } catch (err) {
          console.error('failed to parse generation event', err);
        }
      });
      source.onerror = () => {
        // Ignore errors from a connection that has already been
        // replaced (e.g. by scheduleReconnect or a manual
        // setUrl/unsubscribe). Only the live source matters.
        if (this.source !== sourceRef) return;
        // For both CLOSED and CONNECTING errors, close the live
        // source and route through our backoff path. Without this,
        // the browser's built-in retry on a CONNECTING error
        // bypasses scheduleReconnect and a briefly-flapping backend
        // would be hammered once per second.
        this.disconnect();
        if (this.subscribers.size > 0) {
          this.scheduleReconnect();
        }
      };
      this.source = source;
    } catch (err) {
      console.error('failed to open generation EventSource', err);
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    if (this.subscribers.size === 0) return;
    const delay = this.reconnectDelay;
    // Schedule the next attempt with backoff. disconnect() in the
    // timer callback is intentionally NOT resetting reconnectDelay --
    // only the "ready" listener above (on a successful reconnect) and
    // the last-unsubscriber case in subscribe() reset it. This way a
    // downed backend sees retries at 1s, 2s, 4s, ... up to 30s instead
    // of being hammered once per second.
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, GenerationEventBus.MAX_BACKOFF_MS);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (this.subscribers.size === 0) return;
      this.disconnect();
      this.ensureConnection();
    }, delay);
  }

  private disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.source) {
      try {
        this.source.close();
      } catch {
        // ignore
      }
      this.source = null;
    }
    // Note: we intentionally do NOT reset reconnectDelay here. The
    // "ready" event listener and the last-unsubscriber path reset it.
  }
}

function defaultGenerationEventsUrl(): string {
  // Pull the configured server URL from the zustand store. This keeps
  // the bus URL in sync with whatever apiClient uses (dev 127.0.0.1 vs
  // production Tauri sidecar). Server-side rendering (Next.js etc.)
  // falls back to the relative path, which is fine because the bus
  // never opens there anyway.
  if (typeof window === 'undefined') return '/events/generations';
  try {
    const base = useServerStore.getState().serverUrl;
    return `${base}/events/generations`;
  } catch {
    return '/events/generations';
  }
}

export const generationEventBus = new GenerationEventBus();
