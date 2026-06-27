import { formatDistance } from 'date-fns';
import { ja, zhCN, zhTW, fr } from 'date-fns/locale';
import i18n from '@/i18n';

export function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function getDateLocale() {
  switch (i18n.language) {
    case 'ja':
      return ja;
    case 'zh-CN':
      return zhCN;
    case 'zh-TW':
      return zhTW;
    case 'fr':
      return fr;
    default:
      return undefined;
  }
}

export function formatDate(date: string | Date): string {
  let dateObj: Date;
  if (typeof date === 'string') {
    const dateStr = date.trim();
    if (!dateStr.includes('Z') && !dateStr.match(/[+-]\d{2}:\d{2}$/)) {
      dateObj = new Date(`${dateStr}Z`);
    } else {
      dateObj = new Date(dateStr);
    }
  } else {
    dateObj = date;
  }

  return formatDistance(dateObj, new Date(), {
    addSuffix: true,
    locale: getDateLocale(),
  }).replace(/^about /i, '');
}

export function formatAbsoluteDate(date: string | Date): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  return dateObj.toLocaleString(i18n.language, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

const ENGINE_DISPLAY_NAMES: Record<string, string> = {
  qwen: 'Qwen',
  luxtts: 'LuxTTS',
  chatterbox: 'Chatterbox',
  chatterbox_turbo: 'Chatterbox Turbo',
};

export function formatEngineName(engine?: string, modelSize?: string): string {
  const name = ENGINE_DISPLAY_NAMES[engine ?? 'qwen'] ?? engine ?? 'Qwen';
  if (engine === 'qwen' && modelSize) {
    return `${name} ${modelSize}`;
  }
  return name;
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${Math.round((bytes / k ** i) * 100) / 100} ${sizes[i]}`;
}
