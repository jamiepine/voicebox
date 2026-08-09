import { HttpResponse } from 'msw';

export interface SseEvent {
  data: unknown;
  event?: string;
}

const SSE_HEADERS = {
  'Content-Type': 'text/event-stream',
  'Cache-Control': 'no-cache',
  Connection: 'keep-alive',
} as const;

const encoder = new TextEncoder();

function frame({ data, event }: SseEvent): Uint8Array {
  const payload = typeof data === 'string' ? data : JSON.stringify(data);
  const lines = event ? `event: ${event}\ndata: ${payload}\n\n` : `data: ${payload}\n\n`;
  return encoder.encode(lines);
}

/**
 * An MSW response streaming the given events immediately, then staying open
 * (EventSource reconnects on close, so a closed stream would loop the test).
 */
export function sseResponse(events: SseEvent[]): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const event of events) controller.enqueue(frame(event));
    },
  });
  return new HttpResponse(stream, { headers: SSE_HEADERS });
}

export interface SseController {
  /** Hand this to an MSW resolver: `http.get(url, () => sse.response())`. */
  response(): Response;
  /** Push one event to every open stream. */
  push(event: SseEvent): void;
  /** End all open streams. */
  close(): void;
}

/**
 * Imperative SSE feed for tests that interleave user actions with server
 * events (generation progress, download progress). Each call to `response()`
 * opens a stream that receives subsequent `push`es — matching EventSource
 * reconnect behavior.
 */
export function sseController(): SseController {
  const controllers = new Set<ReadableStreamDefaultController<Uint8Array>>();

  return {
    response() {
      let own: ReadableStreamDefaultController<Uint8Array>;
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          own = controller;
          controllers.add(controller);
        },
        cancel() {
          controllers.delete(own);
        },
      });
      return new HttpResponse(stream, { headers: SSE_HEADERS });
    },
    push(event: SseEvent) {
      for (const controller of controllers) controller.enqueue(frame(event));
    },
    close() {
      for (const controller of controllers) {
        try {
          controller.close();
        } catch {
          // already closed by cancel
        }
      }
      controllers.clear();
    },
  };
}
