import { formatDistance } from 'date-fns';

export function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function formatDate(date: string | Date): string {
  // Parse the date string - if it doesn't have timezone info, treat it as UTC
  let dateObj: Date;
  if (typeof date === 'string') {
    // If the string doesn't end with Z or have timezone offset, assume it's UTC
    const dateStr = date.trim();
    if (!dateStr.includes('Z') && !dateStr.match(/[+-]\d{2}:\d{2}$/)) {
      // No timezone info, treat as UTC
      dateObj = new Date(dateStr + 'Z');
    } else {
      dateObj = new Date(dateStr);
    }
  } else {
    dateObj = date;
  }

  return formatDistance(dateObj, new Date(), { addSuffix: true }).replace(/^about /i, '');
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
