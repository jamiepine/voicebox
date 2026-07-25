import { describe, expect, test } from 'bun:test';
import { shouldMountAudioKeepAlive } from '../src/components/AudioPlayer/AudioKeepAlive';

describe('AudioKeepAlive mounting policy', () => {
  test('mounts in the desktop app when the compatibility setting is enabled', () => {
    expect(shouldMountAudioKeepAlive(true, true)).toBe(true);
  });

  test('does not mount when the user disables the compatibility setting', () => {
    expect(shouldMountAudioKeepAlive(true, false)).toBe(false);
  });

  test('does not mount in the web app', () => {
    expect(shouldMountAudioKeepAlive(false, true)).toBe(false);
  });
});
