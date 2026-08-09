import { expect, test } from '../fixtures';
import { seedProfile } from '../helpers/api';

test('generate speech end to end through the fake TTS pipeline', async ({ page, backend }) => {
  const profile = await seedProfile(backend.url, 'Narrator');

  await page.goto('/');

  // Select the seeded voice, type into the generate box, and submit.
  await page.getByText(profile.name).first().click();
  const input = page.getByRole('textbox').first();
  await input.click();
  await page.keyboard.type('The quick brown fox jumps over the lazy dog.');
  await page.getByRole('button', { name: 'Generate speech' }).click();

  // The row lands in history and completes via the real queue + SSE.
  await expect(page.getByText('The quick brown fox', { exact: false }).first()).toBeVisible({
    timeout: 15_000,
  });

  await expect
    .poll(
      async () => {
        const res = await fetch(`${backend.url}/history?limit=10`);
        const body = (await res.json()) as { items: { status: string }[] };
        return body.items[0]?.status;
      },
      { timeout: 20_000 },
    )
    .toBe('completed');
});
