import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

/** Represents an enumerated audio input microphone device with its ID and display label. */
export interface AudioInputDevice {
  deviceId: string;
  label: string;
}

/**
 * Custom React hook that enumerates available audio input devices (microphones),
 * formats human-readable device labels, and attaches a listener for hardware 'devicechange'
 * events to automatically refresh the input list when microphones are plugged or unplugged.
 *
 * @returns Object containing the array of enumerated audio input devices, a manual refresh function, and loading state.
 */
export function useAudioInputDevices() {
  const { t } = useTranslation();
  const [devices, setDevices] = useState<AudioInputDevice[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const refreshDevices = useCallback(async () => {
    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.enumerateDevices) {
      setDevices([]);
      setIsLoading(false);
      return;
    }

    try {
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = allDevices.filter(
        (d) =>
          d.kind === 'audioinput' &&
          Boolean(d.deviceId) &&
          d.deviceId !== 'default' &&
          d.deviceId !== 'communications',
      );

      let fallbackIndex = 1;
      const formattedDevices: AudioInputDevice[] = audioInputs.map((device) => {
        let label = device.label?.trim();
        if (!label) {
          label = t('settings.captures.dictation.inputDevice.fallbackLabel', {
            index: fallbackIndex,
            defaultValue: `Microphone ${fallbackIndex}`,
          });
          fallbackIndex++;
        }
        return {
          deviceId: device.deviceId,
          label,
        };
      });

      setDevices(formattedDevices);
    } catch (err) {
      console.error('Failed to enumerate audio input devices:', err);
      setDevices([]);
    } finally {
      setIsLoading(false);
    }
  }, [t]);

  useEffect(() => {
    refreshDevices();

    if (typeof navigator !== 'undefined' && navigator.mediaDevices) {
      const mediaDevices = navigator.mediaDevices;
      mediaDevices.addEventListener('devicechange', refreshDevices);
      return () => {
        mediaDevices.removeEventListener('devicechange', refreshDevices);
      };
    }
  }, [refreshDevices]);

  return {
    devices,
    refreshDevices,
    isLoading,
  };
}
