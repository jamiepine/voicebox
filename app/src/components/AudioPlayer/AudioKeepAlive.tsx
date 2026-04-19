import { useEffect, useRef } from 'react';
import { debug } from '@/lib/utils/debug';

// WKWebView tears down the app's CoreAudio output when idle for long enough,
// and a JS-level reload (cmd+R) does NOT restore it — only relaunching the
// Tauri app does. Keeping a silent <audio> element looping forever prevents
// the OS audio session from ever going dormant.
//
// Real silence (zero PCM samples) at full volume is preferred over a muted
// element: browsers/WebKit can optimize muted media away, which defeats the
// purpose of holding the session open.

function buildSilentWavUrl(seconds = 1, sampleRate = 8000): string {
  const numSamples = seconds * sampleRate;
  const bytes = 44 + numSamples * 2;
  const buffer = new ArrayBuffer(bytes);
  const view = new DataView(buffer);
  const write = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  write(0, 'RIFF');
  view.setUint32(4, bytes - 8, true);
  write(8, 'WAVE');
  write(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, 'data');
  view.setUint32(40, numSamples * 2, true);
  return URL.createObjectURL(new Blob([buffer], { type: 'audio/wav' }));
}

export function AudioKeepAlive() {
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    const url = buildSilentWavUrl(1, 8000);
    const el = new Audio(url);
    el.loop = true;
    el.volume = 1;
    el.preload = 'auto';
    audioRef.current = el;

    const tryPlay = () => {
      if (!audioRef.current) return;
      if (!audioRef.current.paused) return;
      audioRef.current.play().catch((err) => {
        debug.log('[AudioKeepAlive] play blocked (will retry on next gesture):', err);
      });
    };

    tryPlay();

    // Autoplay may be blocked until first user interaction — re-attempt then.
    const onGesture = () => tryPlay();
    window.addEventListener('pointerdown', onGesture, { once: false });
    window.addEventListener('keydown', onGesture, { once: false });

    // If the webview ever pauses the element on background, resume on return.
    const onWake = () => {
      if (!document.hidden) tryPlay();
    };
    document.addEventListener('visibilitychange', onWake);
    window.addEventListener('focus', onWake);
    window.addEventListener('pageshow', onWake);

    return () => {
      window.removeEventListener('pointerdown', onGesture);
      window.removeEventListener('keydown', onGesture);
      document.removeEventListener('visibilitychange', onWake);
      window.removeEventListener('focus', onWake);
      window.removeEventListener('pageshow', onWake);
      el.pause();
      el.src = '';
      URL.revokeObjectURL(url);
      audioRef.current = null;
    };
  }, []);

  return null;
}
