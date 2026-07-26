/* Inline stroke icons.
 *
 * Emoji render differently on every platform and carry colour we don't want
 * in a monochrome UI. These are 24x24 stroke paths on a shared grid, so they
 * inherit currentColor and stay optically consistent at any size.
 */

const ICON_PATHS = {
  grid:      '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  server:    '<rect x="3" y="4" width="18" height="7" rx="2"/><rect x="3" y="13" width="18" height="7" rx="2"/><path d="M7 7.5h.01M7 16.5h.01"/>',
  alert:     '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>',
  refresh:   '<path d="M21 12a9 9 0 1 1-2.6-6.4"/><path d="M21 3v6h-6"/>',
  logout:    '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><path d="m16 17 5-5-5-5"/><path d="M21 12H9"/>',
  users:     '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.9"/><path d="M16 3.1a4 4 0 0 1 0 7.8"/>',
  ban:       '<circle cx="12" cy="12" r="9"/><path d="m5.6 5.6 12.8 12.8"/>',
  boot:      '<path d="M18 6 6 18M6 6l12 12"/>',
  clock:     '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  flag:      '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V4s-1 1-4 1-5-2-8-2-4 1-4 1Z"/><path d="M4 22v-7"/>',
  shield:    '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/>',
  trend:     '<path d="m22 7-8.5 8.5-5-5L2 17"/><path d="M16 7h6v6"/>',
  activity:  '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
  message:   '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2Z"/>',
  mic:       '<rect x="9" y="2" width="6" height="11" rx="3"/><path d="M19 10a7 7 0 0 1-14 0"/><path d="M12 17v5"/>',
  ticket:    '<path d="M2 9a3 3 0 0 1 0 6v3a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-3a3 3 0 0 1 0-6V6a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2Z"/><path d="M13 5v2M13 11v2M13 17v2"/>',
  star:      '<path d="m12 3 2.9 5.9 6.5.9-4.7 4.6 1.1 6.5L12 17.8 6.2 20.9l1.1-6.5L2.6 9.8l6.5-.9Z"/>',
  bolt:      '<path d="M13 2 4.1 12.9a1 1 0 0 0 .8 1.6H11l-1 7.5 8.9-10.9a1 1 0 0 0-.8-1.6H13Z"/>',
  search:    '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
  chevron:   '<path d="m9 18 6-6-6-6"/>',
  back:      '<path d="M19 12H5"/><path d="m12 19-7-7 7-7"/>',
  check:     '<path d="M20 6 9 17l-5-5"/>',
  dot:       '<circle cx="12" cy="12" r="4"/>',
  hash:      '<path d="M4 9h16M4 15h16M10 3 8 21M16 3l-2 18"/>',
  brain:     '<path d="M12 5a3 3 0 0 0-6 0 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 6 1Z"/><path d="M12 5a3 3 0 0 1 6 0 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-6 1Z"/>',
  file:      '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/>',
};

/** Render an icon. `size` is px; stroke scales with it. */
function icon(name, size = 16, extraClass = '') {
  const body = ICON_PATHS[name];
  if (!body) return '';
  return `<svg class="icon ${extraClass}" width="${size}" height="${size}" viewBox="0 0 24 24"
    fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"
    stroke-linejoin="round" aria-hidden="true">${body}</svg>`;
}
