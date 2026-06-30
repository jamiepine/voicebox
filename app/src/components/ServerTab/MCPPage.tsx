import { Check, Copy, Plug, Trash2, Waypoints } from 'lucide-react';
import { useState } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useMCPBindings } from '@/lib/hooks/useMCPBindings';
import { useProfiles } from '@/lib/hooks/useProfiles';
import { useCaptureSettings } from '@/lib/hooks/useSettings';
import { useServerStore } from '@/stores/serverStore';
import { formatDate } from '@/lib/utils/format';
import { SettingRow, SettingSection } from './SettingRow';

function getStdioShimCommand(): string {
  if (typeof navigator === 'undefined') {
    return '/Applications/Voicebox.app/Contents/MacOS/voicebox-mcp';
  }

  const platform = `${navigator.platform} ${navigator.userAgent}`.toLowerCase();
  if (platform.includes('win')) {
    return 'C:\\Program Files\\Voicebox\\voicebox-mcp.exe';
  }
  if (platform.includes('linux')) {
    return '/opt/voicebox/voicebox-mcp';
  }
  return '/Applications/Voicebox.app/Contents/MacOS/voicebox-mcp';
}

/**
 * Settings → MCP — configure per-agent voice binding and show copy-paste
 * install snippets for major MCP clients. Backend runs at /mcp on the
 * existing Voicebox server; this page is the agent-onboarding surface.
 */
export function MCPPage() {
  const { t } = useTranslation();
  const serverUrl = useServerStore((s) => s.serverUrl);
  const { bindings, upsertAsync, remove } = useMCPBindings();
  const { data: profiles } = useProfiles();
  const { settings: captureSettings, update: updateCapture } = useCaptureSettings();

  const defaultProfileId = captureSettings?.default_playback_voice_id ?? '';
  const mcpUrl = `${serverUrl}/mcp`;
  const stdioShimCommand = getStdioShimCommand();

  const [newClientId, setNewClientId] = useState('');
  const [newLabel, setNewLabel] = useState('');
  const [newProfileId, setNewProfileId] = useState('');
  const [adding, setAdding] = useState(false);

  const handleAdd = async () => {
    if (!newClientId.trim()) return;
    setAdding(true);
    try {
      await upsertAsync({
        client_id: newClientId.trim(),
        label: newLabel.trim() || null,
        profile_id: newProfileId || null,
      });
      setNewClientId('');
      setNewLabel('');
      setNewProfileId('');
    } finally {
      setAdding(false);
    }
  };

  return (
    <div className="flex gap-8 items-start max-w-5xl">
      <div className="flex-1 min-w-0 max-w-2xl space-y-8">
        <SettingSection
          title={t('settings.mcp.install.title')}
          description={t('settings.mcp.install.description')}
        >
          <SnippetRow
            title={t('settings.mcp.install.http.title')}
            description={t('settings.mcp.install.http.description')}
            snippet={JSON.stringify(
              {
                mcpServers: {
                  voicebox: {
                    url: mcpUrl,
                    headers: { 'X-Voicebox-Client-Id': 'claude-code' },
                  },
                },
              },
              null,
              2,
            )}
          />
          <SnippetRow
            title={t('settings.mcp.install.claudeCode.title')}
            description={t('settings.mcp.install.claudeCode.description')}
            snippet={`claude mcp add voicebox --transport http --url ${mcpUrl} --header "X-Voicebox-Client-Id: claude-code"`}
          />
          <SnippetRow
            title={t('settings.mcp.install.stdio.title')}
            description={t('settings.mcp.install.stdio.description')}
            snippet={JSON.stringify(
              {
                mcpServers: {
                  voicebox: {
                    command: stdioShimCommand,
                    env: { VOICEBOX_CLIENT_ID: 'claude-code' },
                  },
                },
              },
              null,
              2,
            )}
          />
        </SettingSection>

        <SettingSection
          title={t('settings.mcp.defaultVoice.title')}
          description={t('settings.mcp.defaultVoice.description')}
        >
          <SettingRow
            title={t('settings.mcp.defaultVoice.label')}
            description={t('settings.mcp.defaultVoice.labelHint')}
            action={
              <Select
                value={defaultProfileId || '__default__'}
                onValueChange={(v) =>
                  updateCapture({
                    default_playback_voice_id: v === '__default__' ? null : v,
                  })
                }
              >
                <SelectTrigger className="w-[220px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__default__">
                    {t('settings.mcp.defaultVoice.none')}
                  </SelectItem>
                  {(profiles ?? []).map((p) => (
                    <SelectItem key={p.id} value={p.id}>
                      {p.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            }
          />
        </SettingSection>

        <SettingSection
          title={t('settings.mcp.bindings.title')}
          description={t('settings.mcp.bindings.description')}
        >
          {bindings.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4 italic">
              <Trans i18nKey="settings.mcp.bindings.empty" components={{ code: <code /> }} />
            </p>
          ) : (
            <div className="divide-y divide-border/60">
              {bindings.map((b) => (
                <div
                  key={b.client_id}
                  className="py-3 grid grid-cols-[1fr_auto_auto] gap-4 items-center"
                >
                  <div className="min-w-0">
                    <div className="font-medium text-sm truncate">
                      {b.label || b.client_id}
                    </div>
                    <div className="text-xs text-muted-foreground truncate">
                      <code className="text-[11px]">{b.client_id}</code>
                      {' · '}
                      {b.last_seen_at ? (
                        <span title={t('settings.mcp.bindings.lastSeenTitle', { when: b.last_seen_at })}>
                          <Plug className="inline h-3 w-3 text-emerald-500" />{' '}
                          {t('settings.mcp.bindings.lastSeen', { when: formatDate(b.last_seen_at) })}
                        </span>
                      ) : (
                        <span>{t('settings.mcp.bindings.neverConnected')}</span>
                      )}
                    </div>
                  </div>
                  <Select
                    value={b.profile_id ?? '__default__'}
                    onValueChange={(v) =>
                      upsertAsync({
                        client_id: b.client_id,
                        label: b.label,
                        profile_id: v === '__default__' ? null : v,
                      })
                    }
                  >
                    <SelectTrigger className="w-[180px]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="__default__">
                        {t('settings.mcp.bindings.defaultOption')}
                      </SelectItem>
                      {(profiles ?? []).map((p) => (
                        <SelectItem key={p.id} value={p.id}>
                          {p.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    size="icon"
                    variant="ghost"
                    onClick={() => remove(b.client_id)}
                    aria-label={t('settings.mcp.bindings.removeAria', { client: b.client_id })}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
          )}

          <div className="pt-4 space-y-2">
            <div className="text-sm font-medium">{t('settings.mcp.bindings.add.title')}</div>
            <div className="grid grid-cols-[1fr_1fr_auto] gap-2">
              <input
                type="text"
                placeholder={t('settings.mcp.bindings.add.clientIdPlaceholder')}
                value={newClientId}
                onChange={(e) => setNewClientId(e.target.value)}
                className="h-9 px-3 rounded-md border bg-background text-sm"
              />
              <input
                type="text"
                placeholder={t('settings.mcp.bindings.add.labelPlaceholder')}
                value={newLabel}
                onChange={(e) => setNewLabel(e.target.value)}
                className="h-9 px-3 rounded-md border bg-background text-sm"
              />
              <Select
                value={newProfileId || '__default__'}
                onValueChange={(v) => setNewProfileId(v === '__default__' ? '' : v)}
              >
                <SelectTrigger className="h-9 min-w-[140px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__default__">
                    {t('settings.mcp.bindings.defaultOption')}
                  </SelectItem>
                  {(profiles ?? []).map((p) => (
                    <SelectItem key={p.id} value={p.id}>
                      {p.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button
              size="sm"
              onClick={handleAdd}
              disabled={!newClientId.trim() || adding}
            >
              {t('settings.mcp.bindings.add.action')}
            </Button>
          </div>
        </SettingSection>
      </div>

      <aside className="hidden lg:block w-[280px] shrink-0 space-y-6 sticky top-0">
        <div className="space-y-2">
          <h3 className="text-sm font-semibold">{t('settings.mcp.sidebar.aboutTitle')}</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {t('settings.mcp.sidebar.aboutBody')}
          </p>
        </div>
        <div className="space-y-2">
          <h3 className="text-sm font-semibold">{t('settings.mcp.sidebar.toolsTitle')}</h3>
          <ul className="text-sm text-muted-foreground space-y-1.5 leading-relaxed">
            <li>
              <code className="text-accent">voicebox.speak</code>
              <div>{t('settings.mcp.sidebar.tools.speak')}</div>
            </li>
            <li>
              <code className="text-accent">voicebox.transcribe</code>
              <div>{t('settings.mcp.sidebar.tools.transcribe')}</div>
            </li>
            <li>
              <code className="text-accent">voicebox.list_captures</code>
              <div>{t('settings.mcp.sidebar.tools.listCaptures')}</div>
            </li>
            <li>
              <code className="text-accent">voicebox.list_profiles</code>
              <div>{t('settings.mcp.sidebar.tools.listProfiles')}</div>
            </li>
          </ul>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Waypoints className="h-3.5 w-3.5 text-accent" />
          <span>
            <Trans i18nKey="settings.mcp.sidebar.postSpeak" components={{ code: <code /> }} />
          </span>
        </div>
      </aside>
    </div>
  );
}

function SnippetRow({
  title,
  description,
  snippet,
}: {
  title: string;
  description: string;
  snippet: string;
}) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(snippet);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // ignore; user can still select-and-copy the pre content
    }
  };

  return (
    <div className="py-3 space-y-2">
      <div className="flex items-center justify-between gap-4">
        <div>
          <div className="text-sm font-medium">{title}</div>
          <div className="text-xs text-muted-foreground">{description}</div>
        </div>
        <Button size="sm" variant="outline" onClick={copy}>
          {copied ? (
            <>
              <Check className="h-3.5 w-3.5 mr-1.5" />
              {t('settings.mcp.install.copied')}
            </>
          ) : (
            <>
              <Copy className="h-3.5 w-3.5 mr-1.5" />
              {t('settings.mcp.install.copy')}
            </>
          )}
        </Button>
      </div>
      <pre className="text-[11px] font-mono p-3 rounded-md bg-muted/50 overflow-x-auto whitespace-pre-wrap break-all">
        {snippet}
      </pre>
    </div>
  );
}
