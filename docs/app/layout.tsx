import type { ReactNode } from 'react';

// Root layout is a pass-through; the real <html>/<body> live in app/[lang]/layout.tsx.
export default function RootLayout({ children }: { children: ReactNode }) {
  return children;
}
