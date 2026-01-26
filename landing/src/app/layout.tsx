import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Header } from '@/components/Header';
import { Footer } from '@/components/Footer';

const inter = Inter({ subsets: ['latin'], variable: '--font-sans' });

export const metadata: Metadata = {
  title: 'voicebox - Professional Voice Cloning Desktop App',
  description:
    'Near-perfect voice cloning powered by Qwen3-TTS. Desktop app for Mac, Windows, and Linux. Multi-sample support, smart caching, local or remote inference.',
  keywords: ['voice cloning', 'TTS', 'Qwen3', 'desktop app', 'AI voice'],
  icons: {
    icon: [
      { url: '/favicon.ico', sizes: 'any' },
      { url: '/favicon.png', type: 'image/png' },
    ],
    apple: [
      { url: '/apple-touch-icon.png', sizes: '180x180', type: 'image/png' },
    ],
  },
  openGraph: {
    title: 'voicebox',
    description: 'Professional voice cloning with Qwen3-TTS',
    type: 'website',
    url: 'https://voicebox.sh',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className="dark">
      <body className={inter.variable}>
        <div className="relative min-h-screen bg-background font-sans flex flex-col">
          <Header />
          <main className="container mx-auto px-4 sm:px-6 md:px-4 flex-1 py-4 sm:py-6 md:py-0">
            {children}
          </main>
          <Footer />
        </div>
      </body>
    </html>
  );
}
