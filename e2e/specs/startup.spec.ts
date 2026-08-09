import { expect, test } from '../fixtures';

test('app boots against the backend and shows the main editor', async ({ page }) => {
  await page.goto('/');

  await expect(page.getByRole('heading', { name: 'Voicebox' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Create Voice' }).first()).toBeVisible();
});

test('settings layout renders its pages', async ({ page }) => {
  await page.goto('/settings/about');

  await expect(page.getByText('General', { exact: true })).toBeVisible();
  await expect(page.getByText('Changelog', { exact: true })).toBeVisible();
  await expect(page.getByText('About', { exact: true })).toBeVisible();
});
