# Transcripción Whisper de audio largo

## 1. Origen del problema

Los audios de varios minutos se procesaban como una sola inferencia, aunque el
encoder de Whisper está limitado a ventanas de aproximadamente 30 segundos.
El resultado podía contener solo el inicio del audio.

## 2. Causa raíz técnica

La ruta PyTorch de STT no dividía el PCM largo en ventanas antes de llamar a
`model.generate()`. Además, una ventana podía terminar prematuramente si no se
solicitaban timestamps.

## 3. Fix aplicado

El backend divide el audio en ventanas de 30 segundos, une las transcripciones
con espacios, fuerza `return_timestamps=True` y conserva el idioma mediante
`forced_decoder_ids`.

## 4. Verificación

```bash
python -m pytest backend/tests/test_whisper_chunking.py -q
```

La prueba unitaria usa processor/model falsos y cubre audio corto, tres
ventanas para 70 segundos, timestamps y selección de idioma.

La matriz de calidad WER para audio real de 6:43, formatos WAV/FLAC/MP3 y
carga en frío/caliente requiere `pytest -m gpu` con fixtures locales generadas
por `backend/tests/fixtures/generate_fixtures.sh`.

## 5. Estado upstream

La regresión unitaria es apta para CI; la evidencia de WER y latencia queda
pendiente de hardware y modelos descargados.
