# <título del cambio>

## 1. Origen del problema

- Cómo se descubrió.
- Síntomas concretos: mensajes, logs y comportamiento observable.
- Reproducción mínima en la versión sin el fix.

## 2. Causa raíz técnica

- Archivo y línea aproximada.
- Mecanismo que produce el fallo.
- Por qué los tests existentes no lo detectaban.

## 3. Fix aplicado

- Diff conceptual.
- Decisiones tomadas y alternativas descartadas.

## 4. Verificación

- Tests ejecutados, con comandos copiables.
- Salida esperada y salida observada.
- Criterio de regresión: qué volvería a fallar si se revierte el fix.

## 5. Estado upstream

- PR o commit relacionado.
- Notas para el maintainer.
- Riesgos y rollback.
