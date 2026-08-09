import { describe, expect, it, vi } from 'vitest';
import { queryClient } from '@/lib/queryClient';
import { isLoopbackVoiceboxServerUrl, useServerStore } from '@/stores/serverStore';

describe('serverStore', () => {
  it('invalidates all queries when the server url changes', () => {
    const spy = vi.spyOn(queryClient, 'invalidateQueries');

    useServerStore.getState().setServerUrl('http://10.0.0.5:17493');

    expect(useServerStore.getState().serverUrl).toBe('http://10.0.0.5:17493');
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('does not invalidate queries when the url is unchanged', () => {
    const url = useServerStore.getState().serverUrl;
    const spy = vi.spyOn(queryClient, 'invalidateQueries');

    useServerStore.getState().setServerUrl(url);

    expect(spy).not.toHaveBeenCalled();
  });
});

describe('isLoopbackVoiceboxServerUrl', () => {
  it('matches loopback hosts on the voicebox port', () => {
    expect(isLoopbackVoiceboxServerUrl('http://127.0.0.1:17493')).toBe(true);
    expect(isLoopbackVoiceboxServerUrl('http://localhost:17493')).toBe(true);
    expect(isLoopbackVoiceboxServerUrl('http://[::1]:17493')).toBe(true);
  });

  it('rejects other hosts, ports, and junk', () => {
    expect(isLoopbackVoiceboxServerUrl('http://10.0.0.5:17493')).toBe(false);
    expect(isLoopbackVoiceboxServerUrl('http://127.0.0.1:8000')).toBe(false);
    expect(isLoopbackVoiceboxServerUrl('not a url')).toBe(false);
  });
});
