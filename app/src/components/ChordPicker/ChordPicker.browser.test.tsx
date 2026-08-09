import { expect, it, vi } from 'vitest';
import { ChordPicker } from '@/components/ChordPicker/ChordPicker';
import { renderWithProviders } from '@/test/render';

// ChordPicker listens on window in the capture phase and canonicalizes via
// `event.code`, so raw KeyboardEvents give exact control over which physical
// keys the picker sees (userEvent would depend on the host keyboard layout).
function press(code: string) {
  window.dispatchEvent(new KeyboardEvent('keydown', { code, bubbles: true, cancelable: true }));
}

function release(code: string) {
  window.dispatchEvent(new KeyboardEvent('keyup', { code, bubbles: true, cancelable: true }));
}

async function renderPicker(initialKeys: string[] = []) {
  const onSave = vi.fn();
  const onCancel = vi.fn();
  const screen = await renderWithProviders(
    <ChordPicker
      open
      title="Push-to-talk shortcut"
      initialKeys={initialKeys}
      onSave={onSave}
      onCancel={onCancel}
    />,
  );
  return { screen, onSave, onCancel };
}

it('opens empty with save disabled and flags unsupported keys', async () => {
  const { screen } = await renderPicker();

  await expect.element(screen.getByText('Press your shortcut')).toBeVisible();
  await expect.element(screen.getByText('No keys yet')).toBeVisible();
  await expect.element(screen.getByRole('button', { name: 'Save' })).toBeDisabled();

  // NumpadEnter has no canonical chord name — the picker refuses it and
  // stays empty instead of capturing garbage.
  press('NumpadEnter');
  await expect.element(screen.getByText(/isn't supported in chords/)).toBeVisible();
  await expect.element(screen.getByRole('button', { name: 'Save' })).toBeDisabled();
});

it('captures the held keys and saves them after release', async () => {
  const { screen, onSave } = await renderPicker();

  press('KeyJ');
  await expect.element(screen.getByText('Capturing…')).toBeVisible();
  await expect.element(screen.getByText('J', { exact: true })).toBeVisible();

  press('KeyK');
  await expect.element(screen.getByText('K', { exact: true })).toBeVisible();

  // Releasing everything freezes the peak so the user can save hands-free.
  release('KeyK');
  release('KeyJ');
  await expect.element(screen.getByText('Press your shortcut')).toBeVisible();
  await expect.element(screen.getByText('J', { exact: true })).toBeVisible();
  await expect.element(screen.getByText('K', { exact: true })).toBeVisible();

  await screen.getByRole('button', { name: 'Save' }).click();
  expect(onSave).toHaveBeenCalledExactlyOnceWith(['KeyJ', 'KeyK']);
});

it('keeps the peak set when a key is released mid-chord', async () => {
  const { screen, onSave } = await renderPicker();

  press('KeyA');
  press('KeyB');
  press('KeyC');
  await expect.element(screen.getByText('B', { exact: true })).toBeVisible();

  // Mid-chord the display tracks only the currently held keys...
  release('KeyB');
  await expect.element(screen.getByText('B', { exact: true })).not.toBeInTheDocument();
  await expect.element(screen.getByText('A', { exact: true })).toBeVisible();

  // ...but the captured peak still includes the released key.
  release('KeyA');
  release('KeyC');
  await expect.element(screen.getByText('B', { exact: true })).toBeVisible();

  await screen.getByRole('button', { name: 'Save' }).click();
  expect(onSave).toHaveBeenCalledExactlyOnceWith(['KeyA', 'KeyB', 'KeyC']);
});

it('replaces a longer saved chord with a fresh shorter one', async () => {
  const { screen, onSave } = await renderPicker(['KeyA', 'KeyB', 'KeyC']);

  await expect.element(screen.getByText('A', { exact: true })).toBeVisible();

  // The first key of a new sequence resets the peak, so a single key can
  // beat the three-key seed.
  press('KeyZ');
  release('KeyZ');
  await expect.element(screen.getByText('Z', { exact: true })).toBeVisible();
  await expect.element(screen.getByText('A', { exact: true })).not.toBeInTheDocument();

  await screen.getByRole('button', { name: 'Save' }).click();
  expect(onSave).toHaveBeenCalledExactlyOnceWith(['KeyZ']);
});

it('cancel fires the cancel callback and never saves', async () => {
  const { screen, onSave, onCancel } = await renderPicker(['KeyA']);

  press('KeyQ');
  release('KeyQ');
  await screen.getByRole('button', { name: 'Cancel' }).click();

  expect(onCancel).toHaveBeenCalledOnce();
  expect(onSave).not.toHaveBeenCalled();
});
