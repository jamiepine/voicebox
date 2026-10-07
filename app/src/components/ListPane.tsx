import { X } from 'lucide-react';
import type { CSSProperties, ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils/cn';

interface ListPaneProps {
  className?: string;
  children: ReactNode;
}

export function ListPane({ className, children }: ListPaneProps) {
  return (
    <div className={cn('h-full flex flex-col relative overflow-hidden', className)}>
      <div
        className="absolute top-0 right-0 bottom-0 w-px bg-border pointer-events-none z-30"
        style={{
          maskImage: 'linear-gradient(to bottom, transparent 0, black 50px)',
          WebkitMaskImage: 'linear-gradient(to bottom, transparent 0, black 50px)',
        }}
      />
      <div className="absolute top-0 left-0 right-0 h-20 bg-gradient-to-b from-background to-transparent z-10 pointer-events-none" />
      {children}
    </div>
  );
}

interface ListPaneHeaderProps {
  className?: string;
  children: ReactNode;
}

export function ListPaneHeader({ className, children }: ListPaneHeaderProps) {
  return (
    <div className={cn('absolute top-0 left-0 right-0 z-20 px-4', className)}>{children}</div>
  );
}

interface ListPaneTitleRowProps {
  className?: string;
  children: ReactNode;
}

export function ListPaneTitleRow({ className, children }: ListPaneTitleRowProps) {
  return <div className={cn('flex items-center mb-2', className)}>{children}</div>;
}

interface ListPaneTitleProps {
  className?: string;
  children: ReactNode;
}

export function ListPaneTitle({ className, children }: ListPaneTitleProps) {
  return <h2 className={cn('text-2xl px-4 font-bold truncate', className)}>{children}</h2>;
}

interface ListPaneActionsProps {
  className?: string;
  children: ReactNode;
}

export function ListPaneActions({ className, children }: ListPaneActionsProps) {
  return <div className={cn('ml-auto flex items-center gap-2', className)}>{children}</div>;
}

interface ListPaneSearchProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
}

export function ListPaneSearch({ value, onChange, placeholder, className }: ListPaneSearchProps) {
  const { t } = useTranslation();
  return (
    <div className={cn('relative', className)}>
      <Input
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        // Escape clears without reaching for the mouse; the button is the
        // discoverable equivalent.
        onKeyDown={(e) => {
          if (e.key === 'Escape' && value) {
            e.preventDefault();
            onChange('');
          }
        }}
        className={cn(
          'h-9 text-sm rounded-full focus-visible:ring-0 focus-visible:ring-offset-0',
          value && 'pr-9',
        )}
      />
      {value && (
        <button
          type="button"
          onClick={() => onChange('')}
          aria-label={t('common.clearSearch')}
          className="absolute right-1 top-1/2 -translate-y-1/2 rounded-full p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  );
}

interface ListPaneScrollProps {
  className?: string;
  style?: CSSProperties;
  children: ReactNode;
}

export function ListPaneScroll({ className, style, children }: ListPaneScrollProps) {
  return (
    <div
      className={cn('flex-1 overflow-y-auto overflow-x-hidden pt-24', className)}
      style={style}
    >
      {children}
    </div>
  );
}
