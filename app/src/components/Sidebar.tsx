import { Loader2, Settings, Volume2 } from 'lucide-react';
import voiceboxLogo from '@/assets/voicebox-logo.png';
import { cn } from '@/lib/utils/cn';
import { useGenerationStore } from '@/stores/generationStore';
import { usePlayerStore } from '@/stores/playerStore';

interface SidebarProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
  isMacOS?: boolean;
}

const tabs = [
  { id: 'main', icon: Volume2, label: 'Main' },
  { id: 'settings', icon: Settings, label: 'Settings' },
];

export function Sidebar({ activeTab, onTabChange, isMacOS }: SidebarProps) {
  const isGenerating = useGenerationStore((state) => state.isGenerating);
  const audioUrl = usePlayerStore((state) => state.audioUrl);
  const isPlayerVisible = !!audioUrl;

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
          const isActive = activeTab === tab.id;

          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => onTabChange(tab.id)}
              className={cn(
                'w-12 h-12 rounded-full flex items-center justify-center transition-all duration-200',
                'hover:bg-muted/50',
                isActive ? 'bg-muted/50 text-foreground shadow-lg' : 'text-muted-foreground',
              )}
              title={tab.label}
              aria-label={tab.label}
            >
              <Icon className="h-5 w-5" />
            </button>
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
