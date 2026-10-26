import { describe, expect, test } from 'bun:test';
import { isShortDictationError } from '../src/lib/utils/captureErrors';

describe('capture upload error classification', () => {
  for (const message of ['Could not decode audio', 'Audio is empty or corrupt']) {
    test(`preserves file upload error: ${message}`, () => {
      expect(isShortDictationError(message, 'file')).toBe(false);
    });

    test(`preserves recording upload error: ${message}`, () => {
      expect(isShortDictationError(message, 'recording')).toBe(false);
    });

    test(`keeps the existing dictation notice: ${message}`, () => {
      expect(isShortDictationError(message, 'dictation')).toBe(true);
    });
  }

  test('matches decode errors case-insensitively', () => {
    expect(isShortDictationError('COULD NOT DECODE audio', 'dictation')).toBe(true);
  });

  test('does not mask unrelated or empty errors', () => {
    for (const source of ['file', 'recording', 'dictation'] as const) {
      expect(isShortDictationError('Server unavailable', source)).toBe(false);
      expect(isShortDictationError('', source)).toBe(false);
    }
  });
});
