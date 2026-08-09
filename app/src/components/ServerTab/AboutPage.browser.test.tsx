import { expect, it } from 'vitest';
import { AboutPage } from '@/components/ServerTab/AboutPage';
import { createMockPlatform } from '@/test/mockPlatform';
import { renderWithProviders } from '@/test/render';

it('renders and shows the platform version', async () => {
  const platform = createMockPlatform({
    metadata: { getVersion: async () => '9.9.9-test', isTauri: false },
  });

  const screen = await renderWithProviders(<AboutPage />, { platform });

  await expect.element(screen.getByAltText('Voicebox')).toBeVisible();
  await expect.element(screen.getByText('9.9.9-test', { exact: false })).toBeVisible();
});
