# Límite de chunks para TADA

## 1. Origen del problema

Textos largos para TADA podían degenerar en audio de aproximadamente un
segundo o salida incompleta porque el valor general de 800 caracteres era
demasiado alto para el modelo.

## 2. Causa raíz técnica

La petición de generación aplica el valor por defecto global antes de llamar a
la utilidad de TTS chunked. TADA necesita ventanas más pequeñas.

## 3. Fix aplicado

`effective_max_chunk_chars()` fuerza un máximo de 250 caracteres para TADA,
respeta valores explícitos inferiores y deja intactos los demás motores.

## 4. Verificación

```bash
python -m pytest backend/tests/test_tada_chunk_cap.py -q
```

La prueba cubre `None`, el default 800, un valor menor y motores no-TADA.
La duración real de 2.000 caracteres y la ausencia de truncado requieren una
prueba GPU con el modelo TADA descargado.

## 5. Estado upstream

Unit test apto para CI; validación acústica pendiente de hardware/modelo.
