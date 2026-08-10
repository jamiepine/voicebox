# Liberación de modelos después de una generación

## 1. Origen del problema

La rama de VRAM llamaba a `unload_all_models()` desde
`backend/services/generation.py`, pero la función no existía en el registro de
backends. El `ImportError` quedaba oculto por el `except Exception` del bloque
`finally`; por tanto, una generación completada no liberaba los modelos.

## 2. Causa raíz técnica

`run_generation()` importaba un símbolo inexistente en
`backend/backends/__init__.py`. La falta de un test que importara y ejercitara
ese símbolo permitió que el fix pareciera correcto mientras era un no-op.

## 3. Fix aplicado

`unload_all_models()` ahora:

1. Recorre los backends TTS, STT y LLM registrados.
2. Llama a `unload_model()` de cada instancia, sin detenerse si una falla.
3. Elimina los registros para que las fábricas creen instancias nuevas bajo
   demanda.
4. Vacía las cachés CUDA y MPS cuando están disponibles.
5. Deduplica referencias singleton para no descargar dos veces la misma
   instancia.

El `finally` de `run_generation()` conserva el aislamiento del cleanup: un
fallo al descargar no cambia el resultado ya persistido de la generación.

## 4. Verificación

```bash
python -m pytest backend/tests/test_generation_unload.py -q
```

La prueba cubre existencia, descarga de TTS/STT/LLM, continuidad después de un
backend defectuoso, cachés CUDA/MPS y llamadas desde `run_generation()` tanto
con éxito como con excepción.

La ejecución en este checkout requiere instalar las dependencias de
`backend/requirements.txt`; el Python global no tiene SQLAlchemy.

## 5. Estado upstream

Fix local pendiente de revisión y de validación en una máquina con GPU. No se
considera evidencia suficiente para afirmar el umbral de VRAM de 500 MiB sin
la suite GPU real.
