/** Direct REST seeding against the per-worker backend — faster and less
 *  brittle than driving every prerequisite through the UI. */

export interface SeededProfile {
  id: string;
  name: string;
}

/** A 3-second 220 Hz sine WAV (backend requires samples >= 2s). */
export function buildSampleWav(): Blob {
  const sampleRate = 24_000;
  const seconds = 3;
  const samples = sampleRate * seconds;
  const dataSize = samples * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const writeString = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);
  for (let i = 0; i < samples; i++) {
    view.setInt16(44 + i * 2, Math.round(0.1 * 32767 * Math.sin((2 * Math.PI * 220 * i) / sampleRate)), true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

export async function seedProfile(backendUrl: string, name: string): Promise<SeededProfile> {
  const createRes = await fetch(`${backendUrl}/profiles`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, language: 'en' }),
  });
  if (!createRes.ok) throw new Error(`create profile failed: ${createRes.status}`);
  const profile = (await createRes.json()) as { id: string };

  const form = new FormData();
  form.append('file', buildSampleWav(), 'sample.wav');
  form.append('reference_text', 'hello world this is a reference sample');
  const sampleRes = await fetch(`${backendUrl}/profiles/${profile.id}/samples`, {
    method: 'POST',
    body: form,
  });
  if (!sampleRes.ok) throw new Error(`add sample failed: ${sampleRes.status} ${await sampleRes.text()}`);

  return { id: profile.id, name };
}
