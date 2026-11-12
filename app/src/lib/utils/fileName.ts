/**
 * Turn text into something safe to use in a download file name.
 *
 * Letters, combining marks and digits in any script are kept (Arabic, Chinese,
 * Cyrillic, Devanagari, ...); everything else becomes "-". An ASCII-only
 * pattern turns text in those scripts into a file name made only of dashes.
 */
export function toFileNameSlug(text: string, maxLength?: number): string {
  const head = maxLength === undefined ? text : text.substring(0, maxLength);
  return head.replace(/[^\p{L}\p{M}\p{N}]/gu, '-').toLowerCase();
}
