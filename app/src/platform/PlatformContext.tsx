import { createContext, useContext, type ReactNode } from 'react';
import type { Platform } from './types';

const PlatformContext = createContext<Platform | null>(null);

export interface PlatformProviderProps {
  platform: Platform;
  children: ReactNode;
}

export function PlatformProvider({ platform, children }: PlatformProviderProps) {
  return <PlatformContext.Provider value={platform}>{children}</PlatformContext.Provider>;
}

export function usePlatform(): Platform {
  const platform = useContext(PlatformContext);
  if (!platform) {
    throw new Error('usePlatform must be used within PlatformProvider');
  }
  return platform;
}
