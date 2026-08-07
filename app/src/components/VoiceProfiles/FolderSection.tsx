import { ChevronDown, ChevronRight, MoreHorizontal, Pencil, Trash2 } from 'lucide-react';
import { useState } from 'react';
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
import { cn } from '@/lib/utils/cn';
import { isFolderDrag, readFolderDragData } from '@/lib/utils/folderDrag';

interface FolderSectionProps {
  /** Null renders the Uncategorised bucket, which has no menu and no id. */
  folderId: string | null;
  name: string;
  count: number;
  collapsed: boolean;
  onToggle: () => void;
  onRename?: (name: string) => void;
  onDelete?: () => void;
  /** Called when an item is dropped on this header. */
  onDropItem?: (itemId: string) => void;
  children: React.ReactNode;
}

/**
 * A collapsible group header with its members underneath.
 *
 * The delete copy is explicit that only the folder goes — the server
 * releases members to Uncategorised rather than cascading, and a header
 * that just says "Delete" over a group of voices reads far more alarming
 * than what actually happens.
 */
export function FolderSection({
  folderId,
  name,
  count,
  collapsed,
  onToggle,
  onRename,
  onDelete,
  onDropItem,
  children,
}: FolderSectionProps) {
  const { t } = useTranslation();
  const [renameOpen, setRenameOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [draftName, setDraftName] = useState(name);
  const [dragOver, setDragOver] = useState(false);

  const Chevron = collapsed ? ChevronRight : ChevronDown;

  const submitRename = () => {
    const trimmed = draftName.trim();
    if (trimmed && trimmed !== name) onRename?.(trimmed);
    setRenameOpen(false);
  };

  return (
    <div className="flex flex-col gap-1">
      {/* Shaded and bold so the header reads as a container rather than
          blending into the rows it holds. */}
      <div
        onDragOver={(e) => {
          if (!onDropItem || !isFolderDrag(e)) return;
          // preventDefault is what marks this element as a valid drop target.
          e.preventDefault();
          e.dataTransfer.dropEffect = 'move';
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          setDragOver(false);
          const payload = readFolderDragData(e);
          if (!payload || payload.kind !== 'voice') return;
          e.preventDefault();
          onDropItem?.(payload.id);
        }}
        className={cn(
          'group/folder flex items-center gap-1 rounded-md border border-border/60 bg-muted/60 px-1 transition-colors',
          dragOver && 'border-accent bg-accent/40 ring-1 ring-accent',
        )}
      >
        <button
          type="button"
          onClick={onToggle}
          className="flex min-w-0 flex-1 items-center gap-1.5 rounded px-1 py-1.5 text-left hover:bg-accent/30"
          aria-expanded={!collapsed}
        >
          <Chevron className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          <span className="truncate text-xs font-bold uppercase tracking-wide text-foreground">
            {name}
          </span>
          <span className="shrink-0 rounded bg-background/70 px-1 text-[10px] font-semibold tabular-nums text-muted-foreground">
            {count}
          </span>
        </button>

        {folderId && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 shrink-0 opacity-0 transition-opacity focus-visible:opacity-100 group-hover/folder:opacity-100 data-[state=open]:opacity-100"
                aria-label={t('folders.actions', { name })}
              >
                <MoreHorizontal className="h-3.5 w-3.5" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem
                onClick={() => {
                  setDraftName(name);
                  setRenameOpen(true);
                }}
              >
                <Pencil className="mr-2 h-4 w-4" />
                {t('folders.rename')}
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                className="text-destructive focus:text-destructive"
                onClick={() => setDeleteOpen(true)}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                {t('folders.delete')}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>

      {/* Indent and rule the members so they read as belonging to the header
          above rather than as a flat continuation of the list. */}
      {!collapsed && (
        <div className="ml-2 flex flex-col gap-1 border-l border-border/50 pl-2">{children}</div>
      )}

      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t('folders.renameDialog.title')}</DialogTitle>
          </DialogHeader>
          <Input
            value={draftName}
            onChange={(e) => setDraftName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') submitRename();
            }}
            aria-label={t('folders.renameDialog.title')}
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setRenameOpen(false)}>
              {t('common.cancel')}
            </Button>
            <Button onClick={submitRename} disabled={!draftName.trim()}>
              {t('common.save')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t('folders.deleteDialog.title')}</DialogTitle>
            <DialogDescription>{t('folders.deleteDialog.body', { name })}</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteOpen(false)}>
              {t('common.cancel')}
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                onDelete?.();
                setDeleteOpen(false);
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
