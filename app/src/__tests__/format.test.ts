/**
 * Tests for format utility functions.
 *
 * Covers the pure helpers that don't require a browser context.
 * formatDate / formatAbsoluteDate are skipped here because they
 * depend on the i18n singleton and date-fns locale resolution.
 */

import { describe, it, expect } from 'vitest';
import { formatDuration, formatEngineName, formatFileSize } from '../lib/utils/format';

describe('formatDuration', () => {
  it('formats zero seconds', () => {
    expect(formatDuration(0)).toBe('0:00');
  });

  it('formats seconds under a minute', () => {
    expect(formatDuration(30)).toBe('0:30');
  });

  it('pads single-digit seconds', () => {
    expect(formatDuration(61)).toBe('1:01');
  });

  it('formats multi-minute durations', () => {
    expect(formatDuration(125)).toBe('2:05');
  });
});

describe('formatEngineName', () => {
  it('defaults to Qwen when no engine is provided', () => {
    expect(formatEngineName()).toBe('Qwen');
  });

  it('returns Qwen for the qwen engine', () => {
    expect(formatEngineName('qwen')).toBe('Qwen');
  });

  it('appends model size for qwen engine', () => {
    expect(formatEngineName('qwen', '3B')).toBe('Qwen 3B');
  });

  it('does not append model size for other engines', () => {
    expect(formatEngineName('chatterbox', '3B')).toBe('Chatterbox');
  });

  it('formats known engines correctly', () => {
    expect(formatEngineName('luxtts')).toBe('LuxTTS');
    expect(formatEngineName('chatterbox')).toBe('Chatterbox');
    expect(formatEngineName('chatterbox_turbo')).toBe('Chatterbox Turbo');
  });

  it('returns the raw engine string for unknown engines', () => {
    expect(formatEngineName('future_engine')).toBe('future_engine');
  });
});

describe('formatFileSize', () => {
  it('formats zero bytes', () => {
    expect(formatFileSize(0)).toBe('0 Bytes');
  });

  it('formats bytes', () => {
    expect(formatFileSize(512)).toBe('512 Bytes');
  });

  it('formats kilobytes', () => {
    expect(formatFileSize(1024)).toBe('1 KB');
  });

  it('formats megabytes', () => {
    expect(formatFileSize(1024 * 1024)).toBe('1 MB');
  });

  it('formats gigabytes', () => {
    expect(formatFileSize(1024 * 1024 * 1024)).toBe('1 GB');
  });
});
