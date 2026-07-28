import { MessageSquare } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Toggle } from '@/components/ui/toggle';
import { useConversationSettings } from '@/lib/hooks/useSettings';
import { SettingRow, SettingSection } from './SettingRow';

export function ConversationPage() {
  const { settings, update } = useConversationSettings();

  // Local state mirrors the persisted values so inputs don't lag
  const [endpoint, setEndpoint] = useState(settings?.llm_endpoint ?? '');
  const [apiKey, setApiKey] = useState(settings?.llm_api_key ?? '');
  const [model, setModel] = useState(settings?.llm_model ?? '');
  const [systemPrompt, setSystemPrompt] = useState(settings?.system_prompt_prefix ?? '');

  // Sync local state when server data arrives
  useEffect(() => {
    if (settings) {
      setEndpoint(settings.llm_endpoint ?? '');
      setApiKey(settings.llm_api_key ?? '');
      setModel(settings.llm_model ?? '');
      setSystemPrompt(settings.system_prompt_prefix ?? '');
    }
  }, [settings]);

  function commitEndpoint() {
    update({ llm_endpoint: endpoint || null });
  }

  function commitApiKey() {
    update({ llm_api_key: apiKey || null });
  }

  function commitModel() {
    update({ llm_model: model || null });
  }

  function commitSystemPrompt() {
    update({ system_prompt_prefix: systemPrompt || null });
  }

  return (
    <div className="flex gap-8 items-start max-w-5xl">
      <div className="flex-1 min-w-0 max-w-2xl space-y-8">
        <SettingSection
          title="Conversation Mode"
          description="Connect any OpenAI-compatible LLM to power real-time voice conversations in a profile's cloned voice."
        >
          <SettingRow
            title="Enable conversation mode"
            description="When enabled, the Voice Chat panel becomes available in voice profiles."
            htmlFor="conversationEnabled"
            action={
              <Toggle
                id="conversationEnabled"
                checked={settings?.enabled ?? false}
                onCheckedChange={(v) => update({ enabled: v })}
              />
            }
          />

          <SettingRow
            title="LLM endpoint URL"
            description="Base URL of an OpenAI-compatible API (without /v1/chat/completions)."
          >
            <Input
              value={endpoint}
              onChange={(e) => setEndpoint(e.target.value)}
              onBlur={commitEndpoint}
              placeholder="http://localhost:11434/v1"
              className="font-mono text-sm"
            />
          </SettingRow>

          <SettingRow
            title="API key (optional)"
            description="Bearer token for authenticated endpoints. Leave blank for local servers."
          >
            <Input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              onBlur={commitApiKey}
              placeholder="sk-…"
              className="font-mono text-sm"
            />
          </SettingRow>

          <SettingRow
            title="Model name"
            description="The model identifier sent in each request."
          >
            <Input
              value={model}
              onChange={(e) => setModel(e.target.value)}
              onBlur={commitModel}
              placeholder="llama3"
              className="font-mono text-sm"
            />
          </SettingRow>

          <SettingRow
            title="System prompt prefix (optional)"
            description="Prepended before the profile's personality prompt. Use to set global instructions for all conversations."
          >
            <Textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              onBlur={commitSystemPrompt}
              placeholder="You are a helpful AI assistant. Reply concisely and stay in character."
              className="text-sm resize-none"
              rows={4}
            />
          </SettingRow>
        </SettingSection>
      </div>

      <aside className="hidden lg:block w-[280px] shrink-0 space-y-6 sticky top-0">
        <div className="space-y-2">
          <h3 className="text-sm font-semibold">How it works</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Each conversational turn sends the message history to your configured LLM, then
            synthesises the reply using the selected voice profile. Audio streams back via the
            existing TTS pipeline.
          </p>
        </div>

        <div className="space-y-3">
          <h3 className="text-sm font-semibold">Compatible providers</h3>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li className="flex gap-2">
              <MessageSquare className="h-4 w-4 shrink-0 mt-0.5 text-accent" />
              <span>
                <span className="text-foreground font-medium">Ollama</span> — set endpoint to{' '}
                <code className="text-xs bg-muted px-1 rounded">http://localhost:11434/v1</code>
              </span>
            </li>
            <li className="flex gap-2">
              <MessageSquare className="h-4 w-4 shrink-0 mt-0.5 text-accent" />
              <span>
                <span className="text-foreground font-medium">LM Studio</span> — uses its built-in
                OpenAI-compatible server
              </span>
            </li>
            <li className="flex gap-2">
              <MessageSquare className="h-4 w-4 shrink-0 mt-0.5 text-accent" />
              <span>
                <span className="text-foreground font-medium">OpenAI</span> — set endpoint to{' '}
                <code className="text-xs bg-muted px-1 rounded">https://api.openai.com/v1</code>{' '}
                with your API key
              </span>
            </li>
          </ul>
        </div>
      </aside>
    </div>
  );
}
