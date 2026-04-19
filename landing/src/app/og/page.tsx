'use client';

export default function OgPreview() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-[#0a0a09] p-10">
      {/* The card — standard OG image dimensions */}
      <div
        id="og"
        className="relative flex items-center overflow-hidden"
        style={{
          width: 1200,
          height: 630,
          background:
            'radial-gradient(ellipse 80% 70% at 50% 45%, hsla(43,60%,50%,0.12) 0%, hsla(43,60%,50%,0.04) 40%, transparent 70%), linear-gradient(180deg, hsl(30,4%,6%) 0%, hsl(30,4%,4%) 100%)',
        }}
      >
        {/* Logo + text — left-justified, horizontal */}
        <div className="relative z-10 flex items-center" style={{ paddingLeft: 20 }}>
          {/* Glow behind logo */}
          <div
            className="pointer-events-none absolute rounded-full blur-[100px]"
            style={{
              width: 300,
              height: 300,
              top: '50%',
              left: 0,
              transform: 'translateY(-50%)',
              background: 'hsla(43, 60%, 50%, 0.15)',
            }}
          />
          <img
            src="/voicebox-logo-app.webp"
            alt=""
            className="relative shrink-0 object-contain"
            style={{ width: 260, height: 260 }}
            draggable={false}
          />
          <div className="flex flex-col" style={{ marginLeft: -8 }}>
            <h1
              className="font-bold tracking-tight"
              style={{
                fontSize: 72,
                lineHeight: 1,
                color: 'hsl(30, 10%, 94%)',
              }}
            >
              Voicebox
            </h1>
            <p
              style={{
                fontSize: 24,
                lineHeight: 1.4,
                marginTop: 16,
                color: 'hsl(30, 5%, 55%)',
              }}
            >
              Open source voice cloning.
              <br />
              Local-first. Free forever.
            </p>
          </div>
        </div>

        {/* App screenshot — right, overflowing */}
        <img
          src="/assets/app-screenshot-1.webp"
          alt=""
          className="pointer-events-none absolute top-1/2 -translate-y-1/2 z-10"
          style={{
            right: -300,
            width: 900,
          }}
          draggable={false}
        />

        {/* Border overlay */}
        <div className="pointer-events-none absolute inset-0 ring-1 ring-inset ring-white/[0.06]" />
      </div>

      {/* Helper text */}
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 text-xs text-white/30">
        1200 &times; 630 &mdash; Right-click the card or screenshot at 1:1 zoom
      </div>
    </div>
  );
}
