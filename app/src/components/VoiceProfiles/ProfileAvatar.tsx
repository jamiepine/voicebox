import { HugeiconsIcon } from '@hugeicons/react';
import { Mic01Icon } from '@hugeicons/core-free-icons';
import { useState } from 'react';
import { cn } from '@/lib/utils/cn';
import { useServerStore } from '@/stores/serverStore';

interface ProfileAvatarProps {
  profileId: string;
  avatarPath?: string | null;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  grayscale?: boolean;
  className?: string;
  alt?: string;
}

const sizeClasses = {
  sm: 'h-6 w-6',
  md: 'h-8 w-8',
  lg: 'h-10 w-10',
  xl: 'h-24 w-24',
};

const iconSizes = {
  sm: 14,
  md: 16,
  lg: 20,
  xl: 40,
};

const iconClassNames = {
  sm: 'h-3.5 w-3.5',
  md: 'h-4 w-4',
  lg: 'h-5 w-5',
  xl: 'h-10 w-10',
};

export function ProfileAvatar({
  profileId,
  avatarPath,
  size = 'md',
  grayscale = false,
  className,
  alt = 'Profile avatar',
}: ProfileAvatarProps) {
  const [avatarError, setAvatarError] = useState(false);
  const serverUrl = useServerStore((state) => state.serverUrl);

  // If avatarPath is explicitly null or empty string, don't try to load avatar
  // Otherwise, always try to load (avatarPath might not be available in all contexts)
  const avatarUrl =
    avatarPath === null || avatarPath === '' ? null : `${serverUrl}/profiles/${profileId}/avatar`;

  return (
    <div
      className={cn(
        sizeClasses[size],
        'rounded-full bg-muted flex items-center justify-center shrink-0 overflow-hidden',
        className,
      )}
    >
      {avatarUrl && !avatarError ? (
        <img
          src={avatarUrl}
          alt={alt}
          className={cn(
            'h-full w-full object-cover transition-all duration-200',
            grayscale && 'grayscale',
          )}
          onError={() => setAvatarError(true)}
        />
      ) : (
        <HugeiconsIcon
          icon={Mic01Icon}
          size={iconSizes[size]}
          className={cn(iconClassNames[size], 'text-muted-foreground')}
        />
      )}
    </div>
  );
}
