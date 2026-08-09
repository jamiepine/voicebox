import { type HttpHandler, HttpResponse, http } from 'msw';

/**
 * Baseline handlers for endpoints nearly every screen touches. The health
 * payload mirrors backend/routes/health.py closely enough for the UI's
 * checks (`status`, `model_loaded`, backend variant fields).
 */
export const serverHandlers: HttpHandler[] = [
  http.get('*/health', () =>
    HttpResponse.json({
      status: 'ok',
      model_loaded: false,
      device: 'cpu',
      backend_variant: 'cpu',
    }),
  ),
];
