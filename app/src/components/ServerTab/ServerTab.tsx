import { ConnectionForm } from '@/components/ServerSettings/ConnectionForm';
import { GenerationSettings } from '@/components/ServerSettings/GenerationSettings';
import { GpuAcceleration } from '@/components/ServerSettings/GpuAcceleration';
import { UpdateStatus } from '@/components/ServerSettings/UpdateStatus';
import { BOTTOM_SAFE_AREA_PADDING } from '@/lib/constants/ui';
import { cn } from '@/lib/utils/cn';
import { usePlatform } from '@/platform/PlatformContext';
import { usePlayerStore } from '@/stores/playerStore';

export function ServerTab() {
  const platform = usePlatform();
  const isPlayerVisible = !!usePlayerStore((state) => state.audioUrl);
  return (
    <div
      className={cn('overflow-y-auto flex flex-col', isPlayerVisible && BOTTOM_SAFE_AREA_PADDING)}
    >
      <div className="grid gap-4 md:grid-cols-2">
        <ConnectionForm />
        <GenerationSettings />
        {platform.metadata.isTauri && <GpuAcceleration />}
        {platform.metadata.isTauri && <UpdateStatus />}
      </div>
      <div className="py-8 text-center text-sm text-muted-foreground">
        Created by{' '}
        <a
          href="https://github.com/jamiepine"
          target="_blank"
          rel="noopener noreferrer"
          className="text-accent hover:underline"
        >
          Jamie Pine
        </a>
      </div>
    </div>
  );
}
