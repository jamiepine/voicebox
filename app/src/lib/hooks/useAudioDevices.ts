import { useCallback, useEffect, useState } from 'react';

export interface AudioInputDevice {
  deviceId: string;
  label: string;
}

/**
 * Enumerates audio input devices and refreshes on devicechange events.
 * Returns the list of devices and the selected deviceId (persisted to localStorage).
 *
 * NOTE: browsers only expose device labels after getUserMedia has been granted.
 * We request permission lazily — if the list has no labels, we trigger a silent
 * getUserMedia to unlock them, then re-enumerate.
 */
const STORAGE_KEY = 'voicebox:audioInputDeviceId';

function loadStoredDeviceId(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

function saveDeviceId(deviceId: string | null) {
  try {
    if (deviceId) {
      localStorage.setItem(STORAGE_KEY, deviceId);
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  } catch {
    // ignore
  }
}

export function useAudioDevices() {
  const [devices, setDevices] = useState<AudioInputDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceIdState] = useState<string | null>(
    loadStoredDeviceId,
  );

  const enumerateDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;

    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      const inputs = all.filter((d) => d.kind === 'audioinput');

      // If all labels are empty we don't have permission yet — request it silently
      if (inputs.length > 0 && inputs.every((d) => !d.label)) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          stream.getTracks().forEach((t) => t.stop());
          // Re-enumerate now that we have permission
          const all2 = await navigator.mediaDevices.enumerateDevices();
          const inputs2 = all2.filter((d) => d.kind === 'audioinput');
          setDevices(inputs2.map((d) => ({ deviceId: d.deviceId, label: d.label || d.deviceId })));
          return;
        } catch {
          // Permission denied — show unlabeled entries
        }
      }

      setDevices(inputs.map((d) => ({ deviceId: d.deviceId, label: d.label || d.deviceId || 'Unknown device' })));
    } catch (err) {
      console.warn('[useAudioDevices] enumerateDevices failed:', err);
    }
  }, []);

  // Initial enumeration
  useEffect(() => {
    enumerateDevices();
  }, [enumerateDevices]);

  // Refresh list on device plug/unplug
  useEffect(() => {
    if (!navigator.mediaDevices?.addEventListener) return;
    navigator.mediaDevices.addEventListener('devicechange', enumerateDevices);
    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', enumerateDevices);
    };
  }, [enumerateDevices]);

  // If the stored device is no longer available, fall back to default
  useEffect(() => {
    if (devices.length === 0) return;
    if (selectedDeviceId && !devices.find((d) => d.deviceId === selectedDeviceId)) {
      console.log('[useAudioDevices] Stored device no longer available, resetting to default');
      setSelectedDeviceIdState(null);
      saveDeviceId(null);
    }
  }, [devices, selectedDeviceId]);

  const setSelectedDeviceId = useCallback((deviceId: string | null) => {
    setSelectedDeviceIdState(deviceId);
    saveDeviceId(deviceId);
  }, []);

  return {
    devices,
    selectedDeviceId,
    setSelectedDeviceId,
    /** The effective deviceId to pass to getUserMedia — null means "system default" */
    effectiveDeviceId: selectedDeviceId ?? undefined,
  };
}
