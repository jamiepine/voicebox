import { ConnectionForm } from '@/components/ServerSettings/ConnectionForm';
import { GpuAcceleration } from '@/components/ServerSettings/GpuAcceleration';
import { ServerStatus } from '@/components/ServerSettings/ServerStatus';
import { UpdateStatus } from '@/components/ServerSettings/UpdateStatus';
import { usePlatform } from '@/platform/PlatformContext';

export function ServerTab() {
  const platform = usePlatform();
  return (
    <div className="space-y-4 overflow-y-auto flex flex-col">
      <div className="grid gap-4 md:grid-cols-2">
        <ConnectionForm />
        <ServerStatus />
      </div>
      {platform.metadata.isTauri && <GpuAcceleration />}
      {platform.metadata.isTauri && <UpdateStatus />}
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
