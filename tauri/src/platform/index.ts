import type { Platform } from '@/platform/types';
import { tauriAudio } from './audio';
import { tauriFilesystem } from './filesystem';
import { tauriLifecycle } from './lifecycle';
import { tauriMetadata } from './metadata';
import { tauriUpdater } from './updater';

export const tauriPlatform: Platform = {
  filesystem: tauriFilesystem,
  updater: tauriUpdater,
  audio: tauriAudio,
  lifecycle: tauriLifecycle,
  metadata: tauriMetadata,
};
