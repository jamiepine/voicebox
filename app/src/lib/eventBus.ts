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
   * production the apiClient.getGenerationEventsUrl() helper picks
   * the right base URL.
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
      source.addEventListener('ready', () => {
        this.reconnectDelay = 1000;
      });
      source.onerror = () => {
        if (this.source && this.source.readyState === EventSource.CLOSED) {
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
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000);
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
    this.reconnectDelay = 1000;
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