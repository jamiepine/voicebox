import { Link, useRouterState } from '@tanstack/react-router';
import { Box, BookOpen, Loader2, Mic, Server, Speaker, Volume2 } from 'lucide-react';
import voiceboxLogo from '@/assets/voicebox-logo.png';
import { cn } from '@/lib/utils/cn';
import { usePlatform } from '@/platform/PlatformContext';
import { useGenerationStore } from '@/stores/generationStore';
import { usePlayerStore } from '@/stores/playerStore';

interface SidebarProps {
  isMacOS?: boolean;
}

const allTabs = [
  { id: 'main', path: '/', icon: Volume2, label: 'Generate', tauriOnly: false },
  { id: 'stories', path: '/stories', icon: BookOpen, label: 'Stories', tauriOnly: false },
  { id: 'voices', path: '/voices', icon: Mic, label: 'Voices', tauriOnly: false },
  { id: 'audio', path: '/audio', icon: Speaker, label: 'Audio', tauriOnly: true },
  { id: 'models', path: '/models', icon: Box, label: 'Models', tauriOnly: false },
  { id: 'server', path: '/server', icon: Server, label: 'Server', tauriOnly: false },
];

export function Sidebar({ isMacOS }: SidebarProps) {
  const { metadata } = usePlatform();
  const isGenerating = useGenerationStore((state) => state.isGenerating);
  const audioUrl = usePlayerStore((state) => state.audioUrl);
  const isPlayerVisible = !!audioUrl;
  const pathname = useRouterState({ select: (state) => state.location.pathname });

  const tabs = allTabs.filter((tab) => !tab.tauriOnly || metadata.isTauri);

  return (
    <div
      className={cn(
        'fixed left-0 top-0 h-full w-20 bg-sidebar border-r border-border flex flex-col items-center py-6 gap-6',
        isMacOS && 'pt-14',
      )}
    >
      {/* Logo */}
      <div className="mb-2">
        <img src={voiceboxLogo} alt="Voicebox" className="w-12 h-12 object-contain" />
      </div>

      {/* Navigation Buttons */}
      <div className="flex flex-col gap-3">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          // For index route, use exact match; for others, use default matching
          const isActive = tab.path === '/'
            ? pathname === '/'
            : pathname.startsWith(tab.path);

          return (
            <Link
              key={tab.id}
              to={tab.path}
              className={cn(
                'w-12 h-12 rounded-full flex items-center justify-center transition-all duration-200',
                'hover:bg-muted/50',
                isActive ? 'bg-muted/50 text-foreground shadow-lg' : 'text-muted-foreground',
              )}
              title={tab.label}
              aria-label={tab.label}
            >
              <Icon className="h-5 w-5" />
            </Link>
          );
        })}
      </div>

      {/* Spacer to push loader to bottom */}
      <div className="flex-1" />

      {/* Generation Loader */}
      {isGenerating && (
        <div
          className={cn(
            'w-full flex items-center justify-center transition-all duration-200',
            isPlayerVisible ? 'mb-[120px]' : 'mb-0',
          )}
        >
          <Loader2 className="h-6 w-6 text-accent animate-spin" />
        </div>
      )}
    </div>
  );
}
