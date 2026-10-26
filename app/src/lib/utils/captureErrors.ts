import type { CaptureSource } from '@/lib/api/types';

export function isShortDictationError(message: string, source: CaptureSource): boolean {
  return (
    source === 'dictation' &&
    (/could not decode/i.test(message) || /empty or corrupt/i.test(message))
  );
}
