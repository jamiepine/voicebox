import type { ReactNode } from 'react';

/**
 * A section header with title and optional description, separated by a border.
 */
export function SettingSection({
  title,
  description,
  children,
}: {
  title?: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <div className="space-y-1">
      {title && <h3 className="text-sm font-medium">{title}</h3>}
      {description && <p className="text-sm text-muted-foreground">{description}</p>}
      <div className={`${title || description ? 'pt-3' : ''} space-y-0 divide-y divide-border/60`}>
        {children}
      </div>
    </div>
  );
}

/**
 * A single settings row: label+description on the left, action on the right.
 * Use for toggles, inputs, buttons, badges — any control type.
 */
export function SettingRow({
  title,
  description,
  htmlFor,
  action,
  children,
}: {
  title: string;
  description?: string;
  htmlFor?: string;
  /** Right-aligned control (checkbox, button, badge, etc.) */
  action?: ReactNode;
  /** Full-width content rendered below the label row (for sliders, inputs, etc.) */
  children?: ReactNode;
}) {
  return (
    <div className="py-3">
      <div className="flex items-center justify-between gap-8">
        <div className="min-w-0">
          <label
            htmlFor={htmlFor}
            className={`text-sm font-medium leading-none select-none ${htmlFor ? 'cursor-pointer' : ''}`}
          >
            {title}
          </label>
          {description && <p className="text-sm text-muted-foreground mt-0.5">{description}</p>}
        </div>
        {action && <div className="shrink-0">{action}</div>}
      </div>
      {children && <div className="mt-3">{children}</div>}
    </div>
  );
}
