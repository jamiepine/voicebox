import type { Platform } from '@/platform/types';
import { webAudio } from './audio';
import { webFilesystem } from './filesystem';
import { webLifecycle } from './lifecycle';
import { webMetadata } from './metadata';
import { webUpdater } from './updater';

export const webPlatform: Platform = {
  filesystem: webFilesystem,
  updater: webUpdater,
  audio: webAudio,
  lifecycle: webLifecycle,
  metadata: webMetadata,
};
