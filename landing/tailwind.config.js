/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ['class'],
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['var(--font-sans)', 'system-ui', 'sans-serif'],
      },
      colors: {
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
          faint: 'hsl(var(--accent-faint))',
          deep: 'hsl(var(--accent-deep))',
          glow: 'hsl(var(--accent-glow))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
        // App surface tokens
        app: {
          DEFAULT: 'hsl(var(--app))',
          box: 'hsl(var(--app-box))',
          darkBox: 'hsl(var(--app-dark-box))',
          darkerBox: 'hsl(var(--app-darker-box))',
          lightBox: 'hsl(var(--app-light-box))',
          line: 'hsl(var(--app-line))',
          button: 'hsl(var(--app-button))',
          hover: 'hsl(var(--app-hover))',
          selected: 'hsl(var(--app-selected))',
        },
        ink: {
          DEFAULT: 'hsl(var(--ink))',
          dull: 'hsl(var(--ink-dull))',
          faint: 'hsl(var(--ink-faint))',
        },
        sidebar: {
          DEFAULT: 'hsl(var(--sidebar))',
          line: 'hsl(var(--sidebar-line))',
        },
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
};
