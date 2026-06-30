import { motion } from 'framer-motion';

import { cn } from '@/lib/utils/cn';

export type AudioBarsMode = 'idle' | 'generating' | 'playing';

interface AudioBarsProps {
  mode: AudioBarsMode;
  className?: string;
  barClassName?: string;
}

export function AudioBars({ mode, className, barClassName }: AudioBarsProps) {
  const activeColor = mode !== 'idle' ? 'bg-accent' : 'bg-muted-foreground/40';
  return (
    <div className={cn('flex items-center gap-[2px] h-5', className)}>
      {[0, 1, 2, 3, 4].map((i) => (
        <motion.div
          key={`${mode}-${i}`}
          className={cn('w-[3px] rounded-full', activeColor, barClassName)}
          animate={
            mode === 'generating'
              ? { height: ['6px', '16px', '6px'] }
              : mode === 'playing'
                ? { height: ['8px', '14px', '4px', '12px', '8px'] }
                : { height: '8px' }
          }
          transition={
            mode === 'generating'
              ? { duration: 0.6, repeat: Infinity, delay: i * 0.08, ease: 'easeInOut' }
              : mode === 'playing'
                ? { duration: 1.2, repeat: Infinity, delay: i * 0.15, ease: 'easeInOut' }
                : { duration: 0.4, ease: 'easeOut' }
          }
        />
      ))}
    </div>
  );
}
