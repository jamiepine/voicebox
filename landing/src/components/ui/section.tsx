import { cn } from '@/lib/utils';
import type { ReactNode } from 'react';

interface SectionProps {
  children: ReactNode;
  className?: string;
  id?: string;
}

export function Section({ children, className, id }: SectionProps) {
  return (
    <section id={id} className={cn('space-y-6 sm:space-y-8', className)}>
      {children}
    </section>
  );
}

export function SectionTitle({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <h2
      className={cn(
        'text-2xl sm:text-3xl md:text-4xl font-bold text-center md:text-left',
        className,
      )}
    >
      {children}
    </h2>
  );
}
