import { useEffect, useState } from 'react';

/**
 * The value, updated only once it has been stable for `delayMs`.
 *
 * For search inputs that drive a server request: without this, every keystroke
 * is a query. Client-side filters over an in-memory list don't need it.
 */
export function useDebouncedValue<T>(value: T, delayMs = 250): T {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);

  return debounced;
}
