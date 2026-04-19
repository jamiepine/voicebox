import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  metadataBase: new URL('https://voicebox.sh'),
  title: 'Voicebox - Open Source Voice Cloning Desktop App',
  description:
    'Near-perfect voice cloning with multiple TTS engines. Desktop app for Mac, Windows, and Linux. Multi-sample support, smart caching, local or remote inference.',
  keywords: [
    'voice cloning',
    'TTS',
    'multi-engine',
    'desktop app',
    'AI voice',
    'open source',
    'text to speech',
  ],
  icons: {
    icon: [
      { url: '/favicon.png', type: 'image/png' },
      { url: '/favicon.ico', sizes: 'any' },
    ],
    apple: [{ url: '/apple-touch-icon.png', sizes: '180x180', type: 'image/png' }],
  },
  openGraph: {
    title: 'Voicebox',
    description: 'Open source voice cloning. Local-first. Free forever.',
    type: 'website',
    url: 'https://voicebox.sh',
    images: [{ url: '/og.webp', width: 1200, height: 630 }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Voicebox',
    description: 'Open source voice cloning. Local-first. Free forever.',
    images: ['/og.webp'],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Caveat:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <div className="relative min-h-screen bg-background font-sans">{children}</div>
      </body>
    </html>
  );
}
