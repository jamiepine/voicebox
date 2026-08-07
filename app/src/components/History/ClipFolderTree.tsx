import {
  ChevronDown,
  ChevronRight,
  FolderPlus,
  Inbox,
  Layers,
  MoreHorizontal,
  Pencil,
  Trash2,
} from 'lucide-react';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';
import type { FolderKind, FolderResponse } from '@/lib/api/types';
import {
  useCreateFolder,
  useDeleteFolder,
  useDetachFolder,
  useFolders,
  useUpdateFolder,
} from '@/lib/hooks/useFolders';
import { cn } from '@/lib/utils/cn';
import { isFolderDrag, readFolderDragData } from '@/lib/utils/folderDrag';
import { useUIStore } from '@/stores/uiStore';

/**
 * What the clip list is currently filtered to.
 *
 * `uncategorised` is its own selection rather than a null folderId, because
 * "no filter" and "clips in no folder" are different requests — the server
 * distinguishes them too.
 */
export type ClipFolderSelection =
  | { kind: 'all' }
  | { kind: 'uncategorised' }
  | { kind: 'folder'; folderId: string };

interface ClipFolderTreeProps {
  selection: ClipFolderSelection;
  onSelect: (selection: ClipFolderSelection) => void;
  /** Which folder kind to manage. Clips and stories share this tree because
   *  both nest and both need the same create/rename/move/delete affordances. */
  kind?: FolderKind;
  /** Heading above the tree. */
  title?: string;
  /** Label for the "no filter" row. */
  allLabel?: string;
  /** Called when an item is dropped on a folder. folderId is null for the
   *  Uncategorised row. */
  onDropItem?: (itemId: string, folderId: string | null) => void;
}

/** Sentinel for highlighting the Uncategorised row, which has no folder id. */
const UNCATEGORISED_DROP_ID = '__uncategorised__';

/** A folder plus its children, built once per folder list change. */
interface TreeNode {
  folder: FolderResponse;
  children: TreeNode[];
}

function buildTree(folders: FolderResponse[]): TreeNode[] {
  const nodes = new Map<string, TreeNode>();
  for (const folder of folders) nodes.set(folder.id, { folder, children: [] });

  const roots: TreeNode[] = [];
  for (const node of nodes.values()) {
    const parentId = node.folder.parent_id;
    // A parent that isn't in the list (deleted concurrently) would otherwise
    // make the node unreachable — surface it at the root instead.
    const parent = parentId ? nodes.get(parentId) : undefined;
    if (parent) parent.children.push(node);
    else roots.push(node);
  }
  return roots;
}

export function ClipFolderTree({
  selection,
  onSelect,
  kind = 'generation',
  title,
  allLabel,
  onDropItem,
}: ClipFolderTreeProps) {
  const { t } = useTranslation();
  const { data: folders } = useFolders(kind);

  const collapsedIds = useUIStore((state) => state.collapsedFolderIds[kind] ?? []);
  const toggleCollapsed = useUIStore((state) => state.toggleFolderCollapsed);

  const createFolder = useCreateFolder(kind);
  const updateFolder = useUpdateFolder(kind);
  const detachFolder = useDetachFolder(kind);
  const deleteFolder = useDeleteFolder(kind);

  const [dialog, setDialog] = useState<
    | { mode: 'create'; parentId: string | null }
    | { mode: 'rename'; folder: FolderResponse }
    | { mode: 'delete'; folder: FolderResponse }
    | null
  >(null);
  const [draftName, setDraftName] = useState('');
  const [dragOverId, setDragOverId] = useState<string | null>(null);

  const tree = useMemo(() => buildTree(folders ?? []), [folders]);

  const openCreate = (parentId: string | null) => {
    setDraftName('');
    setDialog({ mode: 'create', parentId });
  };

  const submitDialog = () => {
    const trimmed = draftName.trim();
    if (!dialog) return;

    if (dialog.mode === 'create' && trimmed) {
      createFolder.mutate({ name: trimmed, parentId: dialog.parentId });
    } else if (dialog.mode === 'rename' && trimmed && trimmed !== dialog.folder.name) {
      updateFolder.mutate({ folderId: dialog.folder.id, data: { name: trimmed } });
    }
    setDialog(null);
  };

  const renderNode = (node: TreeNode, depth: number) => {
    const { folder, children } = node;
    const collapsed = collapsedIds.includes(folder.id);
    const isSelected = selection.kind === 'folder' && selection.folderId === folder.id;
    const Chevron = collapsed ? ChevronRight : ChevronDown;

    return (
      <div key={folder.id}>
        {/* Shaded and bold, matching the voice folders, so a folder never
            reads as just another row in the list. */}
        <div
          onDragOver={(e) => {
            if (!onDropItem || !isFolderDrag(e)) return;
            // preventDefault is what marks this a valid drop target.
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            setDragOverId(folder.id);
          }}
          onDragLeave={() => setDragOverId((id) => (id === folder.id ? null : id))}
          onDrop={(e) => {
            setDragOverId(null);
            const payload = readFolderDragData(e);
            if (!payload || payload.kind !== kind) return;
            e.preventDefault();
            onDropItem?.(payload.id, folder.id);
          }}
          className={cn(
            'group/node my-0.5 flex items-center gap-1 rounded border border-border/60 bg-muted/60 pr-1 transition-colors',
            isSelected && 'border-accent/60 bg-accent/40',
            dragOverId === folder.id && 'border-accent bg-accent/50 ring-1 ring-accent',
          )}
          style={{ marginLeft: `${depth * 12}px` }}
        >
          {children.length > 0 ? (
            <button
              type="button"
              onClick={() => toggleCollapsed(kind, folder.id)}
              className="shrink-0 rounded p-0.5 hover:bg-accent/50"
              aria-label={folder.name}
              aria-expanded={!collapsed}
            >
              <Chevron className="h-3 w-3 text-muted-foreground" />
            </button>
          ) : (
            // Keeps leaf labels aligned with their expandable siblings.
            <span className="w-4 shrink-0" />
          )}

          <button
            type="button"
            onClick={() => onSelect({ kind: 'folder', folderId: folder.id })}
            className="flex min-w-0 flex-1 items-center gap-1.5 py-1 text-left"
          >
            <span className="truncate text-xs font-bold text-foreground">{folder.name}</span>
            <span className="shrink-0 rounded bg-background/70 px-1 text-[10px] font-semibold tabular-nums text-muted-foreground">
              {folder.item_count}
            </span>
          </button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-5 w-5 shrink-0 opacity-0 transition-opacity focus-visible:opacity-100 group-hover/node:opacity-100 data-[state=open]:opacity-100"
                aria-label={t('folders.actions', { name: folder.name })}
              >
                <MoreHorizontal className="h-3 w-3" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => openCreate(folder.id)}>
                <FolderPlus className="mr-2 h-4 w-4" />
                {t('folders.clip.newSubfolder')}
              </DropdownMenuItem>
              <DropdownMenuItem
                onClick={() => {
                  setDraftName(folder.name);
                  setDialog({ mode: 'rename', folder });
                }}
              >
                <Pencil className="mr-2 h-4 w-4" />
                {t('folders.rename')}
              </DropdownMenuItem>
              {folder.parent_id && (
                <DropdownMenuItem onClick={() => detachFolder.mutate(folder.id)}>
                  <Layers className="mr-2 h-4 w-4" />
                  {t('folders.clip.moveToRoot')}
                </DropdownMenuItem>
              )}
              <DropdownMenuSeparator />
              <DropdownMenuItem
                className="text-destructive focus:text-destructive"
                onClick={() => setDialog({ mode: 'delete', folder })}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                {t('folders.delete')}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {!collapsed && children.map((child) => renderNode(child, depth + 1))}
      </div>
    );
  };

  return (
    <div className="flex flex-col gap-0.5 border-b pb-2 mb-2">
      <div className="flex items-center justify-between px-1">
        <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
          {title ?? t('folders.clip.filterTitle')}
        </span>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          onClick={() => openCreate(null)}
          aria-label={t('folders.new')}
        >
          <FolderPlus className="h-3.5 w-3.5" />
        </Button>
      </div>

      <button
        type="button"
        onClick={() => onSelect({ kind: 'all' })}
        className={cn(
          'flex items-center gap-1.5 rounded px-1 py-1 text-left text-xs',
          selection.kind === 'all' ? 'bg-accent/40' : 'hover:bg-accent/20',
        )}
      >
        <Layers className="h-3 w-3 shrink-0 text-muted-foreground" />
        {allLabel ?? t('folders.clip.allClips')}
      </button>

      {tree.map((node) => renderNode(node, 0))}

      <button
        type="button"
        onClick={() => onSelect({ kind: 'uncategorised' })}
        onDragOver={(e) => {
          if (!onDropItem || !isFolderDrag(e)) return;
          e.preventDefault();
          e.dataTransfer.dropEffect = 'move';
          setDragOverId(UNCATEGORISED_DROP_ID);
        }}
        onDragLeave={() => setDragOverId((id) => (id === UNCATEGORISED_DROP_ID ? null : id))}
        onDrop={(e) => {
          setDragOverId(null);
          const payload = readFolderDragData(e);
          if (!payload || payload.kind !== kind) return;
          e.preventDefault();
          onDropItem?.(payload.id, null);
        }}
        className={cn(
          'flex items-center gap-1.5 rounded px-1 py-1 text-left text-xs transition-colors',
          selection.kind === 'uncategorised' ? 'bg-accent/40' : 'hover:bg-accent/20',
          dragOverId === UNCATEGORISED_DROP_ID && 'bg-accent/50 ring-1 ring-accent',
        )}
      >
        <Inbox className="h-3 w-3 shrink-0 text-muted-foreground" />
        {t('folders.uncategorised')}
      </button>

      <Dialog
        open={dialog?.mode === 'create' || dialog?.mode === 'rename'}
        onOpenChange={() => setDialog(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {dialog?.mode === 'rename'
                ? t('folders.renameDialog.title')
                : t('folders.newDialog.title')}
            </DialogTitle>
          </DialogHeader>
          <Input
            value={draftName}
            onChange={(e) => setDraftName(e.target.value)}
            placeholder={t('folders.newDialog.placeholder')}
            onKeyDown={(e) => {
              if (e.key === 'Enter') submitDialog();
            }}
            aria-label={t('folders.newDialog.title')}
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setDialog(null)}>
              {t('common.cancel')}
            </Button>
            <Button onClick={submitDialog} disabled={!draftName.trim()}>
              {dialog?.mode === 'rename' ? t('common.save') : t('common.create')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={dialog?.mode === 'delete'} onOpenChange={() => setDialog(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t('folders.deleteDialog.title')}</DialogTitle>
            <DialogDescription>
              {dialog?.mode === 'delete' &&
                t('folders.deleteDialog.body', { name: dialog.folder.name })}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDialog(null)}>
              {t('common.cancel')}
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                if (dialog?.mode === 'delete') {
                  const deletedId = dialog.folder.id;
                  deleteFolder.mutate(deletedId);
                  // The filter would otherwise point at a folder that no
                  // longer exists, showing a permanently empty list.
                  if (selection.kind === 'folder' && selection.folderId === deletedId) {
                    onSelect({ kind: 'all' });
                  }
                }
                setDialog(null);
              }}
            >
              {t('folders.deleteDialog.confirm')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
