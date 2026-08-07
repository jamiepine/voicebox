import { BookOpen, FolderInput, MoreHorizontal, Pencil, Plus, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { type ClipFolderSelection, ClipFolderTree } from '@/components/History/ClipFolderTree';
import {
  ListPane,
  ListPaneActions,
  ListPaneHeader,
  ListPaneScroll,
  ListPaneSearch,
  ListPaneTitle,
  ListPaneTitleRow,
} from '@/components/ListPane';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Badge } from '@/components/ui/badge';
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
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/components/ui/use-toast';
import { useFolders, useSetStoryFolder } from '@/lib/hooks/useFolders';
import {
  useCreateStory,
  useDeleteStory,
  useStories,
  useStory,
  useUpdateStory,
} from '@/lib/hooks/useStories';
import { cn } from '@/lib/utils/cn';
import { setFolderDragData } from '@/lib/utils/folderDrag';
import { formatDate } from '@/lib/utils/format';
import { useStoryStore } from '@/stores/storyStore';

export function StoryList() {
  const { t } = useTranslation();
  const { data: stories, isLoading } = useStories();
  const [folderSelection, setFolderSelection] = useState<ClipFolderSelection>({ kind: 'all' });
  const { data: storyFolders } = useFolders('story');
  const setStoryFolder = useSetStoryFolder();

  // Selecting a parent folder should show everything beneath it, matching how
  // the clip panel rolls a subtree up. Stories carry only their own folder id,
  // so expand the selection to the whole subtree here.
  const folderIdsInSubtree = useCallback(
    (rootId: string) => {
      const all = storyFolders ?? [];
      const wanted = new Set([rootId]);
      // Repeat until nothing new is added: the list is flat and unordered, so
      // a single pass could miss grandchildren listed before their parents.
      let grew = true;
      while (grew) {
        grew = false;
        for (const f of all) {
          if (f.parent_id && wanted.has(f.parent_id) && !wanted.has(f.id)) {
            wanted.add(f.id);
            grew = true;
          }
        }
      }
      return wanted;
    },
    [storyFolders],
  );
  const selectedStoryId = useStoryStore((state) => state.selectedStoryId);
  const setSelectedStoryId = useStoryStore((state) => state.setSelectedStoryId);
  const trackEditorHeight = useStoryStore((state) => state.trackEditorHeight);
  const { data: selectedStory } = useStory(selectedStoryId);
  const createStory = useCreateStory();
  const updateStory = useUpdateStory();
  const deleteStory = useDeleteStory();
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [editingStory, setEditingStory] = useState<{
    id: string;
    name: string;
    description?: string;
  } | null>(null);
  const [deletingStoryId, setDeletingStoryId] = useState<string | null>(null);
  const [newStoryName, setNewStoryName] = useState('');
  const [newStoryDescription, setNewStoryDescription] = useState('');
  const [search, setSearch] = useState('');
  const { toast } = useToast();

  // Auto-select the first story when the list loads with no selection
  useEffect(() => {
    if (!selectedStoryId && stories && stories.length > 0) {
      setSelectedStoryId(stories[0].id);
    }
  }, [selectedStoryId, stories, setSelectedStoryId]);

  const handleCreateStory = () => {
    if (!newStoryName.trim()) {
      toast({
        title: t('stories.toast.nameRequired'),
        description: t('stories.toast.nameRequiredDescription'),
        variant: 'destructive',
      });
      return;
    }

    createStory.mutate(
      {
        name: newStoryName.trim(),
        description: newStoryDescription.trim() || undefined,
      },
      {
        onSuccess: (story) => {
          setSelectedStoryId(story.id);
          setCreateDialogOpen(false);
          setNewStoryName('');
          setNewStoryDescription('');
          toast({
            title: t('stories.toast.created'),
            description: t('stories.toast.createdDescription', { name: story.name }),
          });
        },
        onError: (error) => {
          toast({
            title: t('stories.toast.createFailed'),
            description: error.message,
            variant: 'destructive',
          });
        },
      },
    );
  };

  const handleEditClick = (story: { id: string; name: string; description?: string }) => {
    setEditingStory(story);
    setNewStoryName(story.name);
    setNewStoryDescription(story.description || '');
    setEditDialogOpen(true);
  };

  const handleUpdateStory = () => {
    if (!editingStory || !newStoryName.trim()) {
      toast({
        title: t('stories.toast.nameRequired'),
        description: t('stories.toast.nameRequiredDescription'),
        variant: 'destructive',
      });
      return;
    }

    updateStory.mutate(
      {
        storyId: editingStory.id,
        data: {
          name: newStoryName.trim(),
          description: newStoryDescription.trim() || undefined,
        },
      },
      {
        onSuccess: () => {
          setEditDialogOpen(false);
          setEditingStory(null);
          setNewStoryName('');
          setNewStoryDescription('');
        },
        onError: (error) => {
          toast({
            title: t('stories.toast.updateFailed'),
            description: error.message,
            variant: 'destructive',
          });
        },
      },
    );
  };

  const handleDeleteClick = (storyId: string) => {
    setDeletingStoryId(storyId);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = () => {
    if (!deletingStoryId) return;

    deleteStory.mutate(deletingStoryId, {
      onSuccess: () => {
        // Clear selection if deleting the currently selected story
        if (selectedStoryId === deletingStoryId) {
          setSelectedStoryId(null);
        }
        setDeleteDialogOpen(false);
        setDeletingStoryId(null);
      },
      onError: (error) => {
        toast({
          title: t('stories.toast.deleteFailed'),
          description: error.message,
          variant: 'destructive',
        });
      },
    });
  };

  const storyList = stories || [];
  const hasTrackEditor = selectedStoryId && selectedStory && selectedStory.items.length > 0;

  const filtered = useMemo(() => {
    // Folder first, then text — the folder is a scope, the search runs inside it.
    let scoped = storyList;
    if (folderSelection.kind === 'uncategorised') {
      scoped = scoped.filter((s) => !s.folder_id);
    } else if (folderSelection.kind === 'folder') {
      const wanted = folderIdsInSubtree(folderSelection.folderId);
      scoped = scoped.filter((s) => s.folder_id && wanted.has(s.folder_id));
    }

    const q = search.trim().toLowerCase();
    if (!q) return scoped;
    return scoped.filter((s) => {
      const name = (s.name || '').toLowerCase();
      const description = (s.description || '').toLowerCase();
      return name.includes(q) || description.includes(q);
    });
  }, [search, storyList, folderSelection, folderIdsInSubtree]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-muted-foreground">{t('stories.loading')}</div>
      </div>
    );
  }

  return (
    <ListPane>
      <ListPaneHeader>
        <ListPaneTitleRow>
          <ListPaneTitle>{t('stories.title')}</ListPaneTitle>
          <ListPaneActions>
            <Button onClick={() => setCreateDialogOpen(true)} size="sm">
              <Plus className="mr-2 h-4 w-4" />
              {t('stories.newStory')}
            </Button>
          </ListPaneActions>
        </ListPaneTitleRow>
        <ListPaneSearch
          value={search}
          onChange={setSearch}
          placeholder={t('stories.searchPlaceholder')}
        />
      </ListPaneHeader>

      <ListPaneScroll
        style={{ paddingBottom: hasTrackEditor ? `${trackEditorHeight + 140}px` : '170px' }}
      >
        {/* Outside the empty-state branches below: filtering to an empty
            folder must not remove the only control that can clear the filter. */}
        {storyList.length > 0 && (
          <div className="px-4">
            <ClipFolderTree
              kind="story"
              title={t('folders.story.filterTitle')}
              allLabel={t('folders.story.allStories')}
              selection={folderSelection}
              onSelect={setFolderSelection}
              onDropItem={(storyId, folderId) => setStoryFolder.mutate({ storyId, folderId })}
            />
          </div>
        )}

        {storyList.length === 0 ? (
          <div className="mx-4 text-center py-12 px-5 border-2 border-dashed border-muted rounded-2xl text-muted-foreground">
            <BookOpen className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="text-sm">{t('stories.empty.title')}</p>
            <p className="text-xs mt-2">{t('stories.empty.hint')}</p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="px-4 py-12 text-center text-sm text-muted-foreground">
            <p>{t('stories.empty.noMatches', { query: search })}</p>
          </div>
        ) : (
          <div className="px-4 pb-6 space-y-1">
            {filtered.map((story) => {
              const storyFolderId = story.folder_id ?? null;
              const isActive = selectedStoryId === story.id;
              return (
                <div
                  key={story.id}
                  draggable
                  onDragStart={(e) => setFolderDragData(e, { kind: 'story', id: story.id })}
                  className="relative group"
                >
                  <button
                    type="button"
                    onClick={() => setSelectedStoryId(story.id)}
                    aria-label={t('stories.row.ariaLabel', {
                      name: story.name,
                      count: story.item_count,
                      updated: formatDate(story.updated_at),
                    })}
                    aria-pressed={isActive}
                    className={cn(
                      'w-full text-left p-3 rounded-lg transition-colors block',
                      isActive
                        ? 'bg-muted/70 border border-border'
                        : 'border border-transparent hover:bg-muted/30',
                    )}
                  >
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className="text-[11px] text-muted-foreground font-medium">
                        {formatDate(story.updated_at)}
                      </span>
                      <div className="flex-1" />
                    </div>
                    <div className="text-[13px] line-clamp-2 leading-snug mb-2">
                      <span className="text-foreground font-medium">{story.name}</span>
                      {story.description ? (
                        <>
                          <span className="mx-1.5 text-muted-foreground/50">·</span>
                          <span className="text-muted-foreground">{story.description}</span>
                        </>
                      ) : null}
                    </div>
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <Badge
                        variant="secondary"
                        className="h-5 px-1.5 text-[10px] gap-1 font-medium bg-muted/60 text-muted-foreground"
                      >
                        {t('stories.row.itemCount', { count: story.item_count })}
                      </Badge>
                    </div>
                  </button>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="absolute top-2 right-2 h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity"
                        onClick={(e) => e.stopPropagation()}
                        aria-label={t('stories.row.actionsLabel', { name: story.name })}
                      >
                        <MoreHorizontal className="h-3.5 w-3.5" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={() => handleEditClick(story)}>
                        <Pencil className="mr-2 h-4 w-4" />
                        {t('common.edit')}
                      </DropdownMenuItem>
                      <DropdownMenuSub>
                        <DropdownMenuSubTrigger>
                          <FolderInput className="mr-2 h-4 w-4" />
                          {t('folders.story.moveTo')}
                        </DropdownMenuSubTrigger>
                        <DropdownMenuSubContent className="max-h-72 overflow-y-auto">
                          <DropdownMenuItem
                            disabled={!storyFolderId}
                            onClick={() =>
                              setStoryFolder.mutate({ storyId: story.id, folderId: null })
                            }
                          >
                            {t('folders.uncategorised')}
                          </DropdownMenuItem>
                          {(storyFolders ?? []).map((folder) => (
                            <DropdownMenuItem
                              key={folder.id}
                              disabled={folder.id === storyFolderId}
                              onClick={() =>
                                setStoryFolder.mutate({ storyId: story.id, folderId: folder.id })
                              }
                            >
                              {folder.name}
                            </DropdownMenuItem>
                          ))}
                          {(storyFolders ?? []).length === 0 && (
                            <DropdownMenuItem disabled>{t('folders.none')}</DropdownMenuItem>
                          )}
                        </DropdownMenuSubContent>
                      </DropdownMenuSub>
                      <DropdownMenuItem
                        onClick={() => handleDeleteClick(story.id)}
                        className="text-destructive focus:text-destructive"
                      >
                        <Trash2 className="mr-2 h-4 w-4" />
                        {t('common.delete')}
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              );
            })}
          </div>
        )}
      </ListPaneScroll>

      {/* Create Story Dialog */}
      <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t('stories.createDialog.title')}</DialogTitle>
            <DialogDescription>{t('stories.createDialog.description')}</DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="story-name">{t('stories.fields.name')}</Label>
              <Input
                id="story-name"
                placeholder={t('stories.fields.namePlaceholder')}
                value={newStoryName}
                onChange={(e) => setNewStoryName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    handleCreateStory();
                  }
                }}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="story-description">{t('stories.fields.descriptionLabel')}</Label>
              <Textarea
                id="story-description"
                placeholder={t('stories.fields.descriptionPlaceholder')}
                value={newStoryDescription}
                onChange={(e) => setNewStoryDescription(e.target.value)}
                rows={3}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateDialogOpen(false)}>
              {t('common.cancel')}
            </Button>
            <Button onClick={handleCreateStory} disabled={createStory.isPending}>
              {createStory.isPending
                ? t('stories.createDialog.creating')
                : t('stories.createDialog.action')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t('stories.editDialog.title')}</DialogTitle>
            <DialogDescription>{t('stories.editDialog.description')}</DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="edit-story-name">{t('stories.fields.name')}</Label>
              <Input
                id="edit-story-name"
                placeholder={t('stories.fields.namePlaceholder')}
                value={newStoryName}
                onChange={(e) => setNewStoryName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    handleUpdateStory();
                  }
                }}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="edit-story-description">{t('stories.fields.descriptionLabel')}</Label>
              <Textarea
                id="edit-story-description"
                placeholder={t('stories.fields.descriptionPlaceholder')}
                value={newStoryDescription}
                onChange={(e) => setNewStoryDescription(e.target.value)}
                rows={3}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditDialogOpen(false)}>
              {t('common.cancel')}
            </Button>
            <Button onClick={handleUpdateStory} disabled={updateStory.isPending}>
              {updateStory.isPending ? t('stories.editDialog.saving') : t('common.save')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t('stories.deleteDialog.title')}</AlertDialogTitle>
            <AlertDialogDescription>{t('stories.deleteDialog.description')}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
            <AlertDialogAction asChild>
              <Button
                onClick={handleDeleteConfirm}
                disabled={deleteStory.isPending}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                {deleteStory.isPending ? t('stories.deleteDialog.deleting') : t('common.delete')}
              </Button>
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </ListPane>
  );
}
