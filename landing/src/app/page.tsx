'use client';

import { Cloud, Code, Cpu, Github, Shield, Zap } from 'lucide-react';
import Image from 'next/image';
import { useEffect, useState } from 'react';
import { AppleIcon, LinuxIcon, WindowsIcon } from '@/components/PlatformIcons';
import { Button } from '@/components/ui/button';
import { Section, SectionTitle } from '@/components/ui/section';
import { DOWNLOAD_LINKS, GITHUB_REPO } from '@/lib/constants';
import type { DownloadLinks } from '@/lib/releases';
import { FeatureCard } from '../components/ui/feature-card';

export default function Home() {
  const [downloadLinks, setDownloadLinks] = useState<DownloadLinks>(DOWNLOAD_LINKS);

  useEffect(() => {
    // Fetch latest release info
    fetch('/api/releases')
      .then((res) => {
        if (!res.ok) {
          throw new Error('Failed to fetch releases');
        }
        return res.json();
      })
      .then((data) => {
        if (data.downloadLinks) {
          setDownloadLinks(data.downloadLinks);
        }
      })
      .catch((error) => {
        console.error('Failed to fetch release info:', error);
        // Keep fallback links (releases page) on error
      });
  }, []);
  const features = [
    {
      title: 'Near-Perfect Voice Cloning',
      description:
        "Powered by Alibaba's Qwen3-TTS model for exceptional voice quality and accuracy.",
      icon: <Zap className="h-6 w-6" />,
    },
    {
      title: 'Multi-Sample Support',
      description:
        'Combine multiple voice samples for higher quality and more natural-sounding results.',
      icon: <Code className="h-6 w-6" />,
    },
    {
      title: 'Smart Caching',
      description: 'Instant re-generation with voice prompt caching. No need to reprocess samples.',
      icon: <Zap className="h-6 w-6" />,
    },
    {
      title: 'Local or Remote',
      description:
        'Run GPU inference locally or connect to a remote machine. One-click server setup.',
      icon: <Cloud className="h-6 w-6" />,
    },
    {
      title: 'Audio Transcription',
      description:
        'Powered by Whisper for accurate speech-to-text. Extract reference text from voice samples automatically.',
      icon: <Shield className="h-6 w-6" />,
    },
    {
      title: 'Cross-Platform',
      description: 'Available for macOS, Windows, and Linux. No Python installation required.',
      icon: <Cpu className="h-6 w-6" />,
    },
  ];

  return (
    <div className="space-y-12 sm:space-y-16 md:space-y-20">
      {/* Hero Section */}
      <section className="relative py-12 sm:py-16 md:py-20 lg:py-24">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative max-w-7xl">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-start">
            {/* Left side - Content */}
            <div className="space-y-6 lg:pr-8">
              <div className="flex lg:justify-start justify-center mb-6">
                <Image
                  src="/voicebox-logo-2.png"
                  alt="Voicebox Logo"
                  width={1024}
                  height={1024}
                  className="w-32 sm:w-40 md:w-48 h-auto"
                  priority
                />
              </div>
              <h1 className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-bold leading-tight text-center lg:text-left">
                Voicebox
              </h1>
              <p className="text-lg sm:text-xl md:text-2xl text-foreground/70 max-w-xl text-center lg:text-left mx-auto lg:mx-0">
                Open source voice cloning powered by Qwen3-TTS. Create natural-sounding speech from
                text with near-perfect voice replication.
              </p>

              {/* Mobile: centered screenshot above download buttons */}
              <div className="flex justify-center lg:hidden my-8">
                <div className="w-full max-w-2xl">
                  <Image
                    src="/VoiceBoxAppScreenshot.webp"
                    alt="Voicebox Application Screenshot"
                    width={1920}
                    height={1080}
                    className="w-full h-auto rounded-lg shadow-lg"
                    priority
                  />
                </div>
              </div>

              {/* Download buttons under left content */}
              <div className="space-y-4 pt-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-2xl">
                  <Button asChild size="lg" className="w-full px-0">
                    <a
                      href={downloadLinks.macArm}
                      download
                      className="flex items-center w-full relative"
                    >
                      <div className="flex items-center gap-2 flex-shrink-0 pl-4">
                        <AppleIcon className="h-5 w-5" />
                        <div className="h-5 w-px bg-border" />
                      </div>
                      <span className="flex-1 text-center px-4">macOS (ARM)</span>
                    </a>
                  </Button>
                  <Button asChild size="lg" className="w-full px-0">
                    <a
                      href={downloadLinks.macIntel}
                      download
                      className="flex items-center w-full relative"
                    >
                      <div className="flex items-center gap-2 flex-shrink-0 pl-4">
                        <AppleIcon className="h-5 w-5" />
                        <div className="h-5 w-px bg-border" />
                      </div>
                      <span className="flex-1 text-center px-4">macOS (Intel)</span>
                    </a>
                  </Button>
                  <Button asChild size="lg" className="w-full px-0">
                    <a
                      href={downloadLinks.windows}
                      download
                      className="flex items-center w-full relative"
                    >
                      <div className="flex items-center gap-2 flex-shrink-0 pl-4">
                        <WindowsIcon className="h-5 w-5" />
                        <div className="h-5 w-px bg-border" />
                      </div>
                      <span className="flex-1 text-center px-4">Windows</span>
                    </a>
                  </Button>
                  <Button asChild size="lg" className="w-full px-0">
                    <a
                      href={downloadLinks.linux}
                      download
                      className="flex items-center w-full relative"
                    >
                      <div className="flex items-center gap-2 flex-shrink-0 pl-4">
                        <LinuxIcon className="h-5 w-5" />
                        <div className="h-5 w-px bg-border" />
                      </div>
                      <span className="flex-1 text-center px-4">Linux</span>
                    </a>
                  </Button>
                </div>
                <Button variant="outline" size="lg" asChild className="w-full max-w-2xl">
                  <a href={GITHUB_REPO} target="_blank" rel="noopener noreferrer">
                    <Github className="h-4 w-4 mr-2" />
                    View on GitHub
                  </a>
                </Button>
              </div>
            </div>

            {/* Desktop: Large screenshot positioned off-screen */}
            <div className="hidden lg:block relative">
              <div className="absolute right-0 top-0 -mt-10 w-[200%] -mr-[100%]">
                <Image
                  src="/VoiceBoxAppScreenshot.webp"
                  alt="Voicebox Application Screenshot"
                  width={1920}
                  height={1080}
                  className="w-full h-auto"
                  priority
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Demo Video Section */}
      <section className="py-12 sm:py-16 md:py-20">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold text-center mb-8 sm:mb-12">
            See it in action...
          </h2>
          <div className="flex justify-center">
            <div className="w-full max-w-5xl">
              <video
                className="w-full h-auto rounded-lg shadow-lg"
                controls
                playsInline
                preload="metadata"
                poster="/VoiceBoxAppScreenshot.webp"
              >
                <source src="/voicebox-demo.webm" type="video/webm" />
                Your browser does not support the video tag.
              </video>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <Section id="features">
        <SectionTitle className="mb-4 text-center">Features</SectionTitle>
        <p className="text-sm text-muted-foreground mb-8 text-center max-w-2xl mx-auto">
          Everything you need for professional voice cloning in a desktop app.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
          {features.map((feature) => (
            <FeatureCard
              key={feature.title}
              title={feature.title}
              description={feature.description}
              icon={feature.icon}
            />
          ))}
        </div>
      </Section>
    </div>
  );
}
