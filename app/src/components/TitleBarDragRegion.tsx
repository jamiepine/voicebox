const isWindows = navigator.userAgent.includes('Windows');

export function TitleBarDragRegion() {
  if (isWindows) return null;

  return <div data-tauri-drag-region className="fixed top-0 start-0 end-0 h-12 z-[9999]" />;
}
