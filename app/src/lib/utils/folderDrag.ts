import type { FolderKind } from '@/lib/api/types';

/**
 * Payload shared by every "drag an item onto a folder" interaction.
 *
 * Uses native HTML5 drag and drop rather than dnd-kit, which the story
 * timeline uses: those lists only need a drop target, not sortable reordering
 * or collision detection, and native DnD keeps the folder headers as plain
 * elements instead of sensor-wrapped ones.
 *
 * The kind travels with the id so a folder can refuse a drop that belongs to a
 * different panel — dragging a voice onto a clip folder should do nothing
 * rather than fail server-side.
 */

const MIME = 'application/x-voicebox-item';

export interface FolderDragPayload {
  kind: FolderKind;
  id: string;
}

export function setFolderDragData(e: React.DragEvent, payload: FolderDragPayload) {
  e.dataTransfer.setData(MIME, JSON.stringify(payload));
  // Some targets only inspect text/plain; harmless and aids debugging.
  e.dataTransfer.setData('text/plain', payload.id);
  e.dataTransfer.effectAllowed = 'move';
}

/** Read a drag payload, or null when the drag isn't one of ours. */
export function readFolderDragData(e: React.DragEvent): FolderDragPayload | null {
  const raw = e.dataTransfer.getData(MIME);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as FolderDragPayload;
    return parsed?.id && parsed?.kind ? parsed : null;
  } catch {
    return null;
  }
}

/**
 * Whether a drag currently in flight is for this folder kind.
 *
 * dragover cannot read dataTransfer contents (the browser withholds them until
 * drop, for security), so accept-or-not is decided from the *type* being
 * present. The kind is re-checked properly on drop.
 */
export function isFolderDrag(e: React.DragEvent): boolean {
  return e.dataTransfer.types.includes(MIME);
}
