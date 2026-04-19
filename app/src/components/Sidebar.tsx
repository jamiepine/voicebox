import { Link, useMatchRoute } from '@tanstack/react-router';
import { AudioLines, Box, Mic, Settings, Speaker, Volume2, Wand2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import voiceboxLogo from '@/assets/voicebox-logo.png';
import { cn } from '@/lib/utils/cn';
import { usePlatform } from '@/platform/PlatformContext';
import type { UpdateStatus } from '@/platform/types';
import { usePlayerStore } from '@/stores/playerStore';
import { version } from '../../package.json';

interface SidebarProps {
  isMacOS?: boolean;
}

const tabs = [
  { id: 'main', path: '/', icon: Volume2, label: 'Generate' },
  { id: 'stories', path: '/stories', icon: AudioLines, label: 'Stories' },
  { id: 'voices', path: '/voices', icon: Mic, label: 'Voices' },
  { id: 'effects', path: '/effects', icon: Wand2, label: 'Effects' },
  { id: 'audio', path: '/audio', icon: Speaker, label: 'Audio' },
  { id: 'models', path: '/models', icon: Box, label: 'Models' },
  { id: 'settings', path: '/settings', icon: Settings, label: 'Settings' },
];

export function Sidebar({ isMacOS }: SidebarProps) {
  const matchRoute = useMatchRoute();
  const isPlayerOpen = !!usePlayerStore((s) => s.audioUrl);
  const platform = usePlatform();

  const [updateStatus, setUpdateStatus] = useState<UpdateStatus>(platform.updater.getStatus());
  useEffect(() => platform.updater.subscribe(setUpdateStatus), [platform.updater]);

  return (
    <div
      className={cn(
        'fixed left-0 top-0 h-full w-20 bg-sidebar border-r border-border flex flex-col items-center py-6 gap-6',
        isMacOS && 'pt-14',
      )}
    >
      {/* Logo */}
      <div className="mb-2">
        <img
          src={voiceboxLogo}
          alt="Voicebox"
          className="w-12 h-12 object-contain"
          style={{
            filter:
              'drop-shadow(0 0 6px hsl(var(--accent) / 0.5)) drop-shadow(0 0 14px hsl(var(--accent) / 0.35)) drop-shadow(0 0 28px hsl(var(--accent) / 0.2))',
          }}
        />
      </div>

      {/* Navigation Buttons */}
      <div className="flex flex-col gap-3">
        {tabs.map((tab, index) => {
          const Icon = tab.icon;
          const isActive =
            tab.path === '/'
              ? matchRoute({ to: '/', fuzzy: false })
              : matchRoute({ to: tab.path, fuzzy: true });

          // Accent fades as buttons get further from the logo
          const accentOpacity = Math.max(0.08, 0.5 - index * 0.07);

          return (
            <Link
              key={tab.id}
              to={tab.path}
              className={cn(
                'relative w-12 h-12 rounded-full flex items-center justify-center transition-all duration-200 overflow-hidden',
                isActive
                  ? 'bg-white/[0.07] text-foreground shadow-lg backdrop-blur-sm border border-white/[0.08]'
                  : 'text-muted-foreground hover:bg-muted/50',
              )}
              title={tab.label}
              aria-label={tab.label}
            >
              {isActive && (
                <div
                  className="absolute inset-0 rounded-full pointer-events-none"
                  style={{
                    maskImage: 'linear-gradient(to bottom, black, transparent 60%)',
                    WebkitMaskImage: 'linear-gradient(to bottom, black, transparent 60%)',
                    border: `1px solid hsl(var(--accent) / ${accentOpacity})`,
                  }}
                />
              )}
              <Icon className="h-5 w-5 relative z-10" />
            </Link>
          );
        })}
      </div>

      {/* Version */}
      <div
        className="mt-auto flex flex-col items-center gap-1.5 transition-all duration-300"
        style={{ paddingBottom: isPlayerOpen ? '7rem' : undefined }}
      >
        <span className="text-[10px] text-muted-foreground/50">v{version}</span>
        {updateStatus.available && (
          <Link
            to="/settings"
            className="text-[9px] font-semibold tracking-wide uppercase px-2 py-0.5 rounded-full bg-accent/15 text-accent hover:bg-accent/25 transition-colors"
          >
            Update
          </Link>
        )}
      </div>
    </div>
  );
}
