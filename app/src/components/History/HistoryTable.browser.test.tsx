import { HttpResponse, http } from 'msw';
import { expect, it, vi } from 'vitest';
import { HistoryTable } from '@/components/History/HistoryTable';
import { usePlayerStore } from '@/stores/playerStore';
import { buildHistoryItem } from '@/test/msw/fixtures';
import { historyHandlers } from '@/test/msw/handlers';
import { worker } from '@/test/msw/worker';
import { renderWithProviders } from '@/test/render';

it('renders history rows with profile names and transcripts', async () => {
  const ada = buildHistoryItem({
    profile_name: 'Ada Lovelace',
    text: 'The analytical engine speaks.',
  });
  const grace = buildHistoryItem({
    profile_name: 'Grace Hopper',
    text: 'A compiler for the spoken word.',
  });
  worker.use(...historyHandlers([ada, grace]));

  const screen = await renderWithProviders(<HistoryTable />);

  await expect.element(screen.getByText('Ada Lovelace')).toBeVisible();
  await expect.element(screen.getByText('Grace Hopper')).toBeVisible();
  await expect
    .element(screen.getByRole('textbox', { name: /Transcript for sample from Ada Lovelace/ }))
    .toHaveValue('The analytical engine speaks.');
  await expect
    .element(screen.getByRole('textbox', { name: /Transcript for sample from Grace Hopper/ }))
    .toHaveValue('A compiler for the spoken word.');
});

it('shows the empty state when there is no history', async () => {
  worker.use(...historyHandlers([]));

  const screen = await renderWithProviders(<HistoryTable />);

  await expect.element(screen.getByText('No voice generations', { exact: false })).toBeVisible();
});

it('loads a clicked row into the player store with auto-play intent', async () => {
  const item = buildHistoryItem({ profile_name: 'Ada Lovelace', text: 'Play me back.' });
  worker.use(...historyHandlers([item]));

  const screen = await renderWithProviders(<HistoryTable />);

  // Click the profile-name cell — the row's mousedown handler ignores clicks
  // that land on the transcript textarea.
  await screen.getByText('Ada Lovelace').click();

  await expect.poll(() => usePlayerStore.getState().audioId).toBe(item.id);
  const player = usePlayerStore.getState();
  expect(player.audioUrl).toContain(`/audio/${item.id}`);
  expect(player.profileId).toBe(item.profile_id);
  expect(player.shouldAutoPlay).toBe(true);
  // isPlaying flips only once the AudioPlayer (not mounted here) starts playback.
  expect(player.isPlaying).toBe(false);
});

it('toggles favorite via POST and reflects the refetched state', async () => {
  const item = buildHistoryItem({ profile_name: 'Ada Lovelace' });
  let favorited = false;
  const favoriteRequests: string[] = [];
  worker.use(
    http.get('*/history', () =>
      HttpResponse.json({ items: [{ ...item, is_favorited: favorited }], total: 1 }),
    ),
    http.post('*/history/:id/favorite', ({ params }) => {
      favoriteRequests.push(params.id as string);
      favorited = true;
      return HttpResponse.json({ is_favorited: favorited });
    }),
  );

  const screen = await renderWithProviders(<HistoryTable />);

  await screen.getByRole('button', { name: 'Favorite' }).click();

  await expect.poll(() => favoriteRequests).toEqual([item.id]);
  // History was invalidated and refetched — the star now reads as favorited.
  await expect.element(screen.getByRole('button', { name: 'Unfavorite' })).toBeVisible();
});

it('deletes a generation after confirming the dialog', async () => {
  const item = buildHistoryItem({ profile_name: 'Ada Lovelace' });
  let items = [item];
  const deleteRequests: string[] = [];
  worker.use(
    http.get('*/history', () => HttpResponse.json({ items, total: items.length })),
    http.delete('*/history/:id', ({ params }) => {
      deleteRequests.push(params.id as string);
      items = items.filter((i) => i.id !== params.id);
      return HttpResponse.json({ status: 'deleted' });
    }),
  );

  const screen = await renderWithProviders(<HistoryTable />);

  await screen.getByRole('button', { name: 'Actions' }).click();
  await screen.getByRole('menuitem', { name: 'Delete' }).click();
  await expect.element(screen.getByText('Delete Generation')).toBeVisible();
  await screen.getByRole('button', { name: 'Delete' }).click();

  await expect.poll(() => deleteRequests).toEqual([item.id]);
  // The refetched (now empty) list replaces the row.
  await expect.element(screen.getByText('No voice generations', { exact: false })).toBeVisible();
});

it('exports audio through platform.filesystem.saveFile', async () => {
  const item = buildHistoryItem({ profile_name: 'Ada Lovelace', text: 'Export me please' });
  worker.use(
    ...historyHandlers([item]),
    http.get(
      '*/history/:id/export-audio',
      () =>
        new HttpResponse(new Blob([new Uint8Array(64)]), {
          headers: { 'Content-Type': 'audio/wav' },
        }),
    ),
  );

  const screen = await renderWithProviders(<HistoryTable />);

  await screen.getByRole('button', { name: 'Actions' }).click();
  await screen.getByRole('menuitem', { name: 'Export Audio' }).click();

  const saveFile = vi.mocked(screen.platform.filesystem.saveFile);
  await expect.poll(() => saveFile.mock.calls.length).toBe(1);
  const [filename, blob, filters] = saveFile.mock.calls[0];
  expect(filename).toBe('export-me-please.wav');
  expect(blob).toBeInstanceOf(Blob);
  expect(filters).toEqual([{ name: 'Audio File', extensions: ['wav'] }]);
});
