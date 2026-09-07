import { BookOpen, Check, Globe, Pencil, Plus, Search, Trash2, Upload, X } from 'lucide-react';
import { useRef, useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { KnowledgeArticle, KnowledgeSearchResult } from '@/lib/api/types';
import {
  useKnowledge,
  useKnowledgeImports,
  useKnowledgeMutations,
} from '@/lib/hooks/useVoiceAgents';

interface KnowledgePanelProps {
  agentId: string;
}

type Draft = { title: string; content: string; tags: string };

const EMPTY: Draft = { title: '', content: '', tags: '' };

function toTags(s: string): string[] {
  return s
    .split(',')
    .map((t) => t.trim())
    .filter(Boolean);
}

export function KnowledgePanel({ agentId }: KnowledgePanelProps) {
  const { toast } = useToast();
  const { data: articles, isLoading } = useKnowledge(agentId);
  const { create, update, remove } = useKnowledgeMutations(agentId);
  const { importUrl, importFile } = useKnowledgeImports(agentId);
  const [draft, setDraft] = useState<Draft>(EMPTY);
  const [editing, setEditing] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState<Draft>(EMPTY);
  const [url, setUrl] = useState('');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<KnowledgeSearchResult[] | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const add = async () => {
    if (!draft.title.trim() || !draft.content.trim()) return;
    await create.mutateAsync({
      title: draft.title.trim(),
      content: draft.content.trim(),
      tags: toTags(draft.tags),
    });
    setDraft(EMPTY);
  };

  const startEdit = (a: KnowledgeArticle) => {
    setEditing(a.id);
    setEditDraft({ title: a.title, content: a.content, tags: a.tags.join(', ') });
  };

  const saveEdit = async () => {
    if (!editing) return;
    await update.mutateAsync({
      articleId: editing,
      data: {
        title: editDraft.title.trim(),
        content: editDraft.content.trim(),
        tags: toTags(editDraft.tags),
      },
    });
    setEditing(null);
  };

  const doImportUrl = async () => {
    if (!url.trim()) return;
    try {
      const rows = await importUrl.mutateAsync({ url: url.trim() });
      toast({ title: `Imported ${rows.length} entr${rows.length === 1 ? 'y' : 'ies'}` });
      setUrl('');
    } catch (err) {
      toast({
        title: 'Import failed',
        description: err instanceof Error ? err.message : String(err),
        variant: 'destructive',
      });
    }
  };

  const doImportFile = async (file: File | undefined) => {
    if (!file) return;
    try {
      const rows = await importFile.mutateAsync({ file });
      toast({
        title: `Imported ${rows.length} entr${rows.length === 1 ? 'y' : 'ies'} from ${file.name}`,
      });
    } catch (err) {
      toast({
        title: 'Import failed',
        description: err instanceof Error ? err.message : String(err),
        variant: 'destructive',
      });
    } finally {
      if (fileRef.current) fileRef.current.value = '';
    }
  };

  const doSearch = async () => {
    if (!query.trim()) {
      setResults(null);
      return;
    }
    try {
      setResults(await apiClient.searchKnowledge(agentId, query.trim()));
    } catch (err) {
      toast({ title: 'Search failed', description: String(err), variant: 'destructive' });
    }
  };

  return (
    <div className="space-y-4">
      <p className="text-xs text-muted-foreground">
        FAQs, pricing, policies, troubleshooting steps. On every turn the agent pulls the most
        relevant entries into its context and answers only from them — it never invents facts.
        Titles and tags weigh more than body text, so name entries the way customers ask.
      </p>

      {/* Import */}
      <div className="flex gap-2 items-center">
        <Globe className="h-4 w-4 text-muted-foreground shrink-0" />
        <Input
          className="h-9"
          placeholder="https://example.com/help — import a page"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && doImportUrl()}
        />
        <Button
          size="sm"
          variant="outline"
          onClick={doImportUrl}
          disabled={!url.trim() || importUrl.isPending}
        >
          Import page
        </Button>
        <input
          ref={fileRef}
          type="file"
          accept=".txt,.md,.html,.htm,.csv,text/*"
          className="hidden"
          onChange={(e) => doImportFile(e.target.files?.[0])}
        />
        <Button
          size="sm"
          variant="outline"
          onClick={() => fileRef.current?.click()}
          disabled={importFile.isPending}
        >
          <Upload className="h-4 w-4" /> File
        </Button>
      </div>

      {/* Retrieval tester */}
      <div className="rounded-lg border border-border p-3 space-y-2">
        <div className="flex gap-2 items-center">
          <Search className="h-4 w-4 text-muted-foreground shrink-0" />
          <Input
            className="h-9"
            placeholder="Test retrieval: type what a customer might say…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && doSearch()}
          />
          <Button size="sm" variant="ghost" onClick={doSearch}>
            Test
          </Button>
        </div>
        {results && (
          <div className="space-y-1">
            {results.length === 0 && (
              <div className="text-xs text-muted-foreground">
                Nothing would be retrieved for that.
              </div>
            )}
            {results.map((r) => (
              <div key={r.article.id} className="text-xs flex gap-2">
                <span className="tabular-nums text-muted-foreground w-10 shrink-0">
                  {r.score.toFixed(1)}
                </span>
                <span className="font-medium">{r.article.title}</span>
                <span className="text-muted-foreground truncate">{r.article.content}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Add */}
      <div className="rounded-lg border border-border p-3 space-y-2">
        <div className="flex gap-2">
          <Input
            className="h-9"
            placeholder="Title — e.g. “Password reset”"
            value={draft.title}
            onChange={(e) => setDraft({ ...draft, title: e.target.value })}
          />
          <Input
            className="h-9 w-60"
            placeholder="tags, comma, separated"
            value={draft.tags}
            onChange={(e) => setDraft({ ...draft, tags: e.target.value })}
          />
        </div>
        <Textarea
          rows={3}
          placeholder="What the agent may say about this. Write it the way you'd want it read aloud."
          value={draft.content}
          onChange={(e) => setDraft({ ...draft, content: e.target.value })}
        />
        <div className="flex justify-end">
          <Button
            size="sm"
            onClick={add}
            disabled={!draft.title.trim() || !draft.content.trim() || create.isPending}
          >
            <Plus className="h-4 w-4" /> Add entry
          </Button>
        </div>
      </div>

      {isLoading ? (
        <div className="text-sm text-muted-foreground">Loading…</div>
      ) : (articles ?? []).length === 0 ? (
        <div className="text-sm text-muted-foreground py-8 text-center rounded-lg border border-dashed border-border">
          <BookOpen className="h-5 w-5 mx-auto mb-2 opacity-40" />
          No knowledge yet. Support and service agents work far better with a few entries.
        </div>
      ) : (
        <div className="space-y-2">
          {(articles ?? []).map((a) =>
            editing === a.id ? (
              <div key={a.id} className="rounded-lg border border-accent/40 p-3 space-y-2">
                <div className="flex gap-2">
                  <Input
                    className="h-9"
                    value={editDraft.title}
                    onChange={(e) => setEditDraft({ ...editDraft, title: e.target.value })}
                  />
                  <Input
                    className="h-9 w-60"
                    value={editDraft.tags}
                    onChange={(e) => setEditDraft({ ...editDraft, tags: e.target.value })}
                  />
                </div>
                <Textarea
                  rows={4}
                  value={editDraft.content}
                  onChange={(e) => setEditDraft({ ...editDraft, content: e.target.value })}
                />
                <div className="flex justify-end gap-1">
                  <Button size="sm" variant="ghost" onClick={() => setEditing(null)}>
                    <X className="h-4 w-4" /> Cancel
                  </Button>
                  <Button size="sm" onClick={saveEdit} disabled={update.isPending}>
                    <Check className="h-4 w-4" /> Save
                  </Button>
                </div>
              </div>
            ) : (
              <div key={a.id} className="rounded-lg border border-border p-3 group">
                <div className="flex items-start gap-2">
                  <div className="min-w-0 flex-1">
                    <div className="font-medium">{a.title}</div>
                    <div className="text-sm text-muted-foreground whitespace-pre-wrap mt-1">
                      {a.content}
                    </div>
                    <div className="flex gap-1 mt-2 flex-wrap items-center">
                      {a.tags.map((t) => (
                        <Badge key={t} variant="outline" className="text-[10px]">
                          {t}
                        </Badge>
                      ))}
                      {a.source && (
                        <span className="text-[10px] text-muted-foreground truncate">
                          from {a.source}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button
                      size="icon"
                      variant="ghost"
                      className="h-8 w-8"
                      onClick={() => startEdit(a)}
                    >
                      <Pencil className="h-4 w-4" />
                    </Button>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="h-8 w-8 text-muted-foreground hover:text-destructive"
                      onClick={() => remove.mutate(a.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            ),
          )}
        </div>
      )}
    </div>
  );
}
