/**
 * Condense a server error for display in a toast.
 *
 * Some backend errors are enormous and mostly noise. The transformers
 * "Unrecognized model" error is ~4.8KB, of which the first sentence carries all
 * the meaning and the remaining 4.7KB is an alphabetical list of every model
 * architecture it knows about. Rendering that in a 420px toast clipped the text
 * at both ends and pushed the close button off-screen.
 *
 * The rule is deliberately generic rather than pattern-matching any one
 * library: keep the head, cut at the most natural boundary available inside the
 * budget, and report how much was dropped so nobody assumes they read it all.
 */

/** Longest head of an oversized error kept for the toast, before the ellipsis.
 *
 * Roughly the first paragraph — enough for a sentence or two of real message.
 * This bounds the *message* rather than the rendered string: `display` is this
 * plus `TRUNCATION_SUFFIX` when something was cut. Spending two of these
 * characters on the ellipsis instead would shorten the message to no purpose,
 * since nothing downstream has a hard character limit — the description box
 * scrolls.
 */
const HEAD_BUDGET = 400;

/** Marks a `display` value as incomplete. Appended after the head. */
const TRUNCATION_SUFFIX = ' …';

/** Below this, condensing is not worth it and the whole message is shown.
 *
 * Deliberately above the budget rather than equal to it. An error of 450
 * characters would otherwise be cut to 400 to save 50 — a worse result than
 * showing all of it, since the description scrolls anyway. The gap also gives
 * the threshold hysteresis instead of flipping between full and truncated
 * around a single character. No Copy action is offered in this range because
 * nothing is being withheld: `display` already holds the entire message.
 */
const MIN_TO_CONDENSE = HEAD_BUDGET + 120;

export interface CondensedError {
  /** What to show in the toast.
   *
   * Byte-identical to `full` whenever nothing is omitted, so a message that
   * fits is never altered. Only an oversized one is rewritten, into a trimmed
   * head followed by an ellipsis.
   */
  display: string;
  /** The original string exactly as received, for copying. Never modified. */
  full: string;
  /** Whether `display` omits part of the message. */
  truncated: boolean;
  /** How many characters `display` leaves out. */
  omitted: number;
}

export function condenseError(raw: string | null | undefined): CondensedError {
  // `full` is what the Copy action hands over, so it stays byte-for-byte what
  // the server sent. The measuring and cutting below works on a trimmed copy
  // instead — surrounding blank space should not count toward the budget or
  // the omitted count — but `text` is never what gets returned as `display`
  // unless the message is actually being shortened.
  const full = raw ?? '';
  const text = full.trim();

  if (text.length <= MIN_TO_CONDENSE) {
    return { display: full, full, truncated: false, omitted: 0 };
  }

  // A traceback's first line is nearly always the message; prefer it whenever
  // it fits, since a newline is a stronger boundary than any punctuation.
  const firstLine = text.split('\n', 1)[0].trim();
  let head =
    firstLine.length > 0 && firstLine.length <= HEAD_BUDGET
      ? firstLine
      : text.slice(0, HEAD_BUDGET);

  if (head.length < text.length && head === text.slice(0, head.length)) {
    // Back off to the last sentence end inside the budget so the text does not
    // stop mid-word. Only accept it if it keeps most of the budget — otherwise
    // a stray early period would throw away usable context.
    const lastStop = Math.max(head.lastIndexOf('. '), head.lastIndexOf('? '));
    if (lastStop > HEAD_BUDGET * 0.4) {
      head = head.slice(0, lastStop + 1);
    }
  }

  head = head.trimEnd();
  const omitted = text.length - head.length;

  // Guard against the boundary search having produced nothing shorter. Nothing
  // is omitted here either, so the message goes back exactly as it arrived.
  if (omitted <= 0) {
    return { display: full, full, truncated: false, omitted: 0 };
  }

  return { display: `${head}${TRUNCATION_SUFFIX}`, full, truncated: true, omitted };
}
