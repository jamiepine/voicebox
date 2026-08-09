import { expect, test } from '../fixtures';
import { seedProfile } from '../helpers/api';

test('a seeded voice profile appears in the voices tab', async ({ page, backend }) => {
  const profile = await seedProfile(backend.url, 'Marcus Aurelius');

  await page.goto('/voices');

  await expect(page.getByText(profile.name).first()).toBeVisible();
});
