export interface TextChunk {
  id: string;
  text: string;
  charCount: number;
  wordCount: number;
}

function normalizeText(text: string): string {
  return text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim();
}

function splitParagraphIntoSentences(paragraph: string): string[] {
  const trimmed = paragraph.trim();
  if (!trimmed) {
    return [];
  }

  const matches = trimmed.match(/[^.!?]+[.!?]+(?:["')\]]+)?|[^.!?]+$/g);
  if (!matches || matches.length === 0) {
    return [trimmed];
  }

  return matches.map((sentence) => sentence.trim()).filter(Boolean);
}

export function chunkText(
  rawText: string,
  targetChunkSize: number,
  maxChunkSize: number,
): TextChunk[] {
  const text = normalizeText(rawText);
  if (!text) {
    return [];
  }

  const safeTarget = Math.max(200, Math.min(targetChunkSize, maxChunkSize));
  const paragraphs = text
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);

  const chunks: string[] = [];
  let current = '';

  const pushCurrent = () => {
    const normalized = current.trim();
    if (!normalized) {
      return;
    }
    chunks.push(normalized);
    current = '';
  };

  for (const paragraph of paragraphs) {
    const sentences = splitParagraphIntoSentences(paragraph);

    for (const sentence of sentences) {
      // Keep sentence integrity. If one sentence exceeds maxChunkSize,
      // keep it as a single oversized chunk and let UI ask for manual edit.
      if (sentence.length > maxChunkSize) {
        pushCurrent();
        chunks.push(sentence);
        continue;
      }

      if (!current) {
        current = sentence;
        continue;
      }

      const candidate = `${current} ${sentence}`;
      if (candidate.length <= safeTarget) {
        current = candidate;
        continue;
      }

      if (candidate.length <= maxChunkSize && current.length < Math.floor(safeTarget * 0.75)) {
        current = candidate;
        continue;
      }

      pushCurrent();
      current = sentence;
    }

    if (current.length >= Math.floor(safeTarget * 0.8)) {
      pushCurrent();
    }
  }

  pushCurrent();

  return chunks.map((chunkTextValue, index) => ({
    id: `chunk-${index + 1}`,
    text: chunkTextValue,
    charCount: chunkTextValue.length,
    wordCount: chunkTextValue.split(/\s+/).filter(Boolean).length,
  }));
}
