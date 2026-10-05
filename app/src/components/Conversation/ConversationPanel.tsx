import { Loader2, Mic, Play, Send } from 'lucide-react';
import { useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { ConversationMessage } from '@/lib/api/types';
import { useConversationSettings } from '@/lib/hooks/useSettings';
import { cn } from '@/lib/utils/cn';
import { usePlayerStore } from '@/stores/playerStore';
import { useServerStore } from '@/stores/serverStore';

interface ConversationPanelProps {
  profileId: string;
  language?: string;
  engine?: string;
  modelSize?: string;
}

export function ConversationPanel({
  profileId,
  language = 'en',
  engine = 'qwen',
  modelSize = '1.7B',
}: ConversationPanelProps) {
  const { settings } = useConversationSettings();
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const { toast } = useToast();
  const serverUrl = useServerStore((state) => state.serverUrl);
  const setAudioUrl = usePlayerStore((state) => state.setAudioUrl);
  const bottomRef = useRef<HTMLDivElement>(null);

  const isEnabled = settings?.enabled ?? false;

  async function sendMessage() {
    const text = input.trim();
    if (!text || isLoading) return;

    const newUserMsg: ConversationMessage = { role: 'user', content: text };
    const updatedHistory = [...messages, newUserMsg];
    setMessages(updatedHistory);
    setInput('');
    setIsLoading(true);

    try {
      const response = await apiClient.conversationTurn({
        profile_id: profileId,
        user_message: text,
        history: messages,
        language,
        engine,
        model_size: modelSize,
      });

      const assistantMsg: ConversationMessage = {
        role: 'assistant',
        content: response.assistant_text,
      };
      setMessages([...updatedHistory, assistantMsg]);

      // Auto-play if generation was started
      if (response.generation_id) {
        playGenerationAudio(response.generation_id);
      }
    } catch (err) {
      toast({
        title: 'Conversation failed',
        description: err instanceof Error ? err.message : 'Unknown error',
        variant: 'destructive',
      });
      // Roll back the user message on error
      setMessages(messages);
    } finally {
      setIsLoading(false);
      setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 50);
    }
  }

  function playGenerationAudio(generationId: string) {
    setPlayingId(generationId);

    // Poll SSE for completion then play audio
    const statusUrl = `${serverUrl}/generate/${generationId}/status`;
    const evtSource = new EventSource(statusUrl);

    evtSource.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data) as { status: string; id: string };
        if (data.status === 'completed') {
          evtSource.close();
          const audioUrl = `${serverUrl}/audio/${generationId}`;
          setAudioUrl(audioUrl);
          setPlayingId(null);
        } else if (data.status === 'failed') {
          evtSource.close();
          setPlayingId(null);
          toast({
            title: 'Audio generation failed',
            description: 'Could not generate voice for the assistant reply.',
            variant: 'destructive',
          });
        }
      } catch {
        evtSource.close();
        setPlayingId(null);
      }
    };

    evtSource.onerror = () => {
      evtSource.close();
      setPlayingId(null);
    };
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  if (!isEnabled) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3 text-center px-6">
        <Mic className="h-8 w-8 text-muted-foreground/40" />
        <p className="text-sm text-muted-foreground">
          Conversation mode is disabled. Enable it in{' '}
          <strong>Settings → Conversation</strong> and configure an LLM endpoint.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Message list */}
      <div className="flex-1 overflow-y-auto space-y-3 p-4">
        {messages.length === 0 && (
          <div className="text-center text-sm text-muted-foreground py-8">
            Start a conversation with this voice profile.
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={cn(
              'flex gap-2 items-start',
              msg.role === 'user' ? 'justify-end' : 'justify-start',
            )}
          >
            {msg.role === 'assistant' && (
              <div className="shrink-0 mt-1">
                <Mic className="h-4 w-4 text-accent" />
              </div>
            )}
            <div
              className={cn(
                'rounded-lg px-3 py-2 max-w-[80%] text-sm',
                msg.role === 'user'
                  ? 'bg-accent text-accent-foreground'
                  : 'bg-muted text-foreground',
              )}
            >
              {msg.content}
            </div>
            {msg.role === 'assistant' && i === messages.length - 1 && (
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7 shrink-0 mt-0.5"
                disabled={playingId !== null}
                onClick={() => {
                  // Find the most recent generation_id via the last assistant turn
                  // The audio will already be playing from auto-play, but this
                  // lets the user replay it
                  const lastReply = messages
                    .slice()
                    .reverse()
                    .find((m) => m.role === 'assistant');
                  if (lastReply) {
                    // Re-trigger by searching history — simplest approach:
                    // we can't easily recover the generation_id from message state,
                    // so we re-send is not desirable. Just show the global player.
                    toast({ title: 'Use the player bar to replay the last audio.' });
                  }
                }}
              >
                {playingId !== null ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Play className="h-3.5 w-3.5" />
                )}
              </Button>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Thinking…
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="shrink-0 border-t p-3 flex gap-2 items-end">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
          className="resize-none min-h-[60px] max-h-[120px] text-sm"
          disabled={isLoading}
          rows={2}
        />
        <Button
          size="icon"
          disabled={!input.trim() || isLoading}
          onClick={sendMessage}
          className="shrink-0"
        >
          {isLoading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </Button>
      </div>
    </div>
  );
}
