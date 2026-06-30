'use client';

import { Check, Copy } from 'lucide-react';
import { useState } from 'react';

/** Compact, copyable contract address — short display, full value to clipboard. */
export function CopyAddress({ address }: { address: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(address);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard unavailable (e.g. insecure context) — silently no-op.
    }
  };

  const short = `${address.slice(0, 4)}…${address.slice(-4)}`;

  return (
    <button
      type="button"
      onClick={handleCopy}
      title={address}
      aria-label={copied ? 'Contract address copied' : 'Copy contract address'}
      className="group inline-flex items-center gap-2 rounded-lg border border-border/60 bg-card/60 px-2.5 py-1.5 font-mono text-xs text-muted-foreground transition-colors hover:border-border hover:text-foreground"
    >
      <span>{short}</span>
      {copied ? (
        <Check className="h-3.5 w-3.5 text-accent" />
      ) : (
        <Copy className="h-3.5 w-3.5 opacity-60 transition-opacity group-hover:opacity-100" />
      )}
    </button>
  );
}
