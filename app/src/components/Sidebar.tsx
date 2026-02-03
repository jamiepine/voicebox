import {
  Book01Icon,
  Mic01Icon,
  PackageIcon,
  ServerStack01Icon,
  SpeakerIcon,
  VolumeHighIcon,
} from '@hugeicons/core-free-icons';
import { HugeiconsIcon } from '@hugeicons/react';
import { Icon } from '@iconify/react';
import { Link, useMatchRoute } from '@tanstack/react-router';
import voiceboxLogo from '@/assets/voicebox-logo.png';
import { cn } from '@/lib/utils/cn';
import { useGenerationStore } from '@/stores/generationStore';
import { usePlayerStore } from '@/stores/playerStore';

interface SidebarProps {
  isMacOS?: boolean;
}

const tabs = [
  { id: 'main', path: '/', icon: VolumeHighIcon, label: 'Generate' },
  { id: 'stories', path: '/stories', icon: Book01Icon, label: 'Stories' },
  { id: 'voices', path: '/voices', icon: Mic01Icon, label: 'Voices' },
  { id: 'audio', path: '/audio', icon: SpeakerIcon, label: 'Audio' },
  { id: 'models', path: '/models', icon: PackageIcon, label: 'Models' },
  { id: 'server', path: '/server', icon: ServerStack01Icon, label: 'Server' },
];

export function Sidebar({ isMacOS }: SidebarProps) {
  const isGenerating = useGenerationStore((state) => state.isGenerating);
  const audioUrl = usePlayerStore((state) => state.audioUrl);
  const isPlayerVisible = !!audioUrl;
  const matchRoute = useMatchRoute();

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
          const isActive =
            tab.path === '/' ? matchRoute({ to: '/' }) : matchRoute({ to: tab.path });

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
              <HugeiconsIcon icon={Icon} size={20} className="h-5 w-5" />
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
          <Icon icon="svg-spinners:ring-resize" className="h-6 w-6 text-accent animate-spin" />
        </div>
      )}
    </div>
  );
}
