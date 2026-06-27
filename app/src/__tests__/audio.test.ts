/**
 * Tests for audio utility functions.
 *
 * These tests cover the pure (non-Web-API) helpers so we can run them in
 * a Node environment without a browser or jsdom.  The Web Audio path inside
 * getAudioDuration is exercised by the integration tests.
 */

import { describe, it, expect } from 'vitest';
import { createAudioUrl, formatAudioDuration } from '../lib/utils/audio';

describe('createAudioUrl', () => {
  it('joins serverUrl and audioId with a slash', () => {
    expect(createAudioUrl('abc-123', 'http://localhost:17493')).toBe(
      'http://localhost:17493/audio/abc-123',
    );
  });

  it('double-slashes when serverUrl has a trailing slash (documents current behavior)', () => {
    // The function does a simple template literal — callers must not add a
    // trailing slash to serverUrl.  This test documents that contract.
    const url = createAudioUrl('xyz', 'http://localhost:17493/');
    expect(url).toBe('http://localhost:17493//audio/xyz');
  });

  it('works with a UUID-style id', () => {
    const id = '550e8400-e29b-41d4-a716-446655440000';
    expect(createAudioUrl(id, 'https://example.com')).toBe(`https://example.com/audio/${id}`);
  });
});

describe('formatAudioDuration', () => {
  it('formats zero seconds', () => {
    expect(formatAudioDuration(0)).toBe('0:00');
  });

  it('formats seconds-only (< 1 minute)', () => {
    expect(formatAudioDuration(45)).toBe('0:45');
  });

  it('formats exactly one minute', () => {
    expect(formatAudioDuration(60)).toBe('1:00');
  });

  it('pads single-digit seconds with a leading zero', () => {
    expect(formatAudioDuration(65)).toBe('1:05');
  });

  it('formats longer durations correctly', () => {
    // 2 minutes 30 seconds
    expect(formatAudioDuration(150)).toBe('2:30');
    // 10 minutes 0 seconds
    expect(formatAudioDuration(600)).toBe('10:00');
  });

  it('truncates fractional seconds (floor)', () => {
    // 1 minute 59.9 seconds → 1:59
    expect(formatAudioDuration(119.9)).toBe('1:59');
  });
});
