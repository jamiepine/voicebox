import { SiApple, SiLinux } from '@icons-pack/react-simple-icons';

// Official brand icons via Simple Icons (apple/linux). Simple Icons drops
// Microsoft's mark due to trademark policy, so the Windows 11 flag is
// inlined from Microsoft's public brand guidance.

export function AppleIcon({ className }: { className?: string }) {
  return <SiApple className={className} color="currentColor" />;
}

export function LinuxIcon({ className }: { className?: string }) {
  return <SiLinux className={className} color="currentColor" />;
}

export function WindowsIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="currentColor"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label="Windows"
    >
      <title>Windows</title>
      <path d="M0 3.449L9.75 2.1v9.451H0m10.949-9.602L24 0v11.4l-13.051.149M0 12.6h9.75v9.451L0 20.699M10.949 12.6H24V24l-12.9-1.801" />
    </svg>
  );
}
