export type Sponsor = {
  name: string;
  url: string;
  logoSrc: string;
  logoAlt?: string;
  tagline?: string;
  /** Set true for solid-black logos that need to render white on the dark theme. */
  invert?: boolean;
};

export const SPONSORS: Sponsor[] = [];
