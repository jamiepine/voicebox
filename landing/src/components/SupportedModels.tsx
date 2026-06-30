'use client';

import {
  Brain,
  Globe,
  Languages,
  type LucideIcon,
  MessageSquare,
  Mic2,
  SlidersHorizontal,
  Sparkles,
  Volume2,
  Zap,
} from 'lucide-react';

type Tag = { icon: LucideIcon; label: string };

type Model = {
  name: string;
  author: string;
  sizes?: string[];
  description: string;
  tags?: Tag[];
};

type ModelGroup = {
  title: string;
  subtitle: string;
  models: Model[];
};

const MODEL_GROUPS: ModelGroup[] = [
  {
    title: 'TTS Engines',
    subtitle: 'Text → speech. Voice cloning, preset voices, and delivery control.',
    models: [
      {
        name: 'Qwen3-TTS',
        author: 'Alibaba',
        sizes: ['1.7B', '0.6B'],
        description:
          'High-quality multilingual cloning with natural prosody. The only engine with delivery instructions — control tone, pace, and emotion with natural language.',
        tags: [
          { icon: Globe, label: '10 langs' },
          { icon: MessageSquare, label: 'Delivery instructions' },
        ],
      },
      {
        name: 'Chatterbox',
        author: 'Resemble AI',
        description:
          'Production-grade voice cloning with the broadest language support. 23 languages with zero-shot cloning and emotion exaggeration control.',
        tags: [{ icon: Languages, label: '23 langs' }],
      },
      {
        name: 'Chatterbox Turbo',
        author: 'Resemble AI',
        sizes: ['350M'],
        description:
          'Lightweight and fast. Supports paralinguistic tags — embed [laugh], [sigh], [gasp] directly in your text for expressive speech.',
        tags: [
          { icon: Zap, label: 'Fast' },
          { icon: MessageSquare, label: '[tag] support' },
        ],
      },
      {
        name: 'LuxTTS',
        author: 'ZipVoice',
        description:
          'Ultra-fast, CPU-friendly cloning at 48kHz. Exceeds 150x realtime on CPU with ~1GB VRAM. The fastest engine for quick iterations.',
        tags: [
          { icon: Zap, label: '150x realtime' },
          { icon: Volume2, label: '48kHz' },
        ],
      },
      {
        name: 'Qwen CustomVoice',
        author: 'Alibaba',
        sizes: ['1.7B', '0.6B'],
        description:
          'Nine premium preset speakers with natural-language style control. "Speak slowly with warmth", "authoritative and clear" — tone and pace adapt.',
        tags: [
          { icon: SlidersHorizontal, label: 'Instruct control' },
          { icon: Globe, label: '10 langs' },
        ],
      },
      {
        name: 'TADA',
        author: 'Hume AI',
        sizes: ['3B', '1B'],
        description:
          'Speech-language model with text-acoustic dual alignment. Built for long-form — 700s+ coherent audio without drift. Multilingual at 3B.',
        tags: [
          { icon: Globe, label: '10 langs' },
          { icon: MessageSquare, label: 'Long-form' },
        ],
      },
      {
        name: 'Kokoro',
        author: 'hexgrad · Apache 2.0',
        sizes: ['82M'],
        description:
          'Tiny 82M-parameter TTS that runs at CPU realtime with negligible VRAM. Pre-built voice styles — pick a voice, type, generate.',
        tags: [
          { icon: Zap, label: 'CPU realtime' },
          { icon: Volume2, label: 'Preset voices' },
        ],
      },
    ],
  },
  {
    title: 'Transcription',
    subtitle: 'Speech → text. Multi-language STT for dictation and captures.',
    models: [
      {
        name: 'Whisper',
        author: 'OpenAI',
        sizes: ['1.5B', '769M', '244M', '74M'],
        description:
          'The default. Mature multilingual ASR across a wide size range — pick Tiny for speed or Large for best accuracy.',
        tags: [{ icon: Languages, label: '99 langs' }],
      },
      {
        name: 'Whisper Turbo',
        author: 'OpenAI',
        sizes: ['809M'],
        description:
          'Pruned Whisper Large v3. Near-best quality at roughly 8x the speed — the right default for real-time dictation.',
        tags: [
          { icon: Languages, label: '99 langs' },
          { icon: Zap, label: '8x faster' },
        ],
      },
    ],
  },
  {
    title: 'Language Models',
    subtitle: 'Transcript refinement, persona replies, and on-device reasoning.',
    models: [
      {
        name: 'Qwen3',
        author: 'Alibaba',
        sizes: ['4B', '1.7B', '0.6B'],
        description:
          'Powers transcript cleanup, persona voice replies, and the voice I/O loop. Shares its runtime with the TTS/STT stack — one model cache, one GPU story.',
        tags: [
          { icon: Sparkles, label: 'Refinement' },
          { icon: Brain, label: 'Persona replies' },
        ],
      },
    ],
  },
];

function ModelCard({ model }: { model: Model }) {
  return (
    <div className="rounded-xl border border-border bg-card/60 backdrop-blur-sm p-4 transition-colors hover:border-accent/30 flex flex-col">
      <div className="flex items-start justify-between gap-2 mb-1.5">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-foreground truncate">{model.name}</h3>
          <span className="text-[11px] text-muted-foreground/60">by {model.author}</span>
        </div>
        {model.sizes && model.sizes.length > 0 && (
          <div className="flex flex-wrap gap-1 justify-end shrink-0 max-w-[55%]">
            {model.sizes.map((s) => (
              <span
                key={s}
                className="text-[9px] px-1.5 py-0.5 rounded-full border border-border bg-background text-muted-foreground whitespace-nowrap tabular-nums"
              >
                {s}
              </span>
            ))}
          </div>
        )}
      </div>
      <p className="text-xs text-muted-foreground leading-relaxed mb-3 flex-1">
        {model.description}
      </p>
      {model.tags && model.tags.length > 0 && (
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          {model.tags.map((tag) => {
            const Icon = tag.icon;
            return (
              <span
                key={tag.label}
                className="flex items-center gap-1 text-[10px] text-muted-foreground/70"
              >
                <Icon className="h-2.5 w-2.5" />
                {tag.label}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ModelGroupSection({ group }: { group: ModelGroup }) {
  return (
    <div>
      {/* Group header */}
      <div className="flex items-baseline justify-between gap-4 mb-5">
        <div>
          <h3 className="text-base font-semibold text-foreground">{group.title}</h3>
          <p className="text-sm text-muted-foreground/80">{group.subtitle}</p>
        </div>
        <span className="text-[11px] font-mono text-ink-faint/60 tabular-nums shrink-0">
          {String(group.models.length).padStart(2, '0')} model
          {group.models.length === 1 ? '' : 's'}
        </span>
      </div>

      {/* Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {group.models.map((model) => (
          <ModelCard key={model.name} model={model} />
        ))}
      </div>
    </div>
  );
}

export function SupportedModels() {
  return (
    <section id="about" className="border-t border-border py-24">
      <div className="mx-auto max-w-6xl px-6">
        <div className="text-center mb-14">
          <h2 className="text-3xl font-semibold tracking-tight text-foreground md:text-4xl mb-4">
            Supported models
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Pick the right model for every job — TTS, transcription, refinement. All models run
            locally on your hardware. Download once, use forever.
          </p>
        </div>

        <div className="space-y-14">
          {MODEL_GROUPS.map((group) => (
            <ModelGroupSection key={group.title} group={group} />
          ))}
        </div>
      </div>
    </section>
  );
}
