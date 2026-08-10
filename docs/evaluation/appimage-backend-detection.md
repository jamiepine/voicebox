# Detección de backend de la AppImage

## 1. Origen del problema

La aplicación debe reutilizar un backend Voicebox ya activo, rechazar un puerto
ocupado por otra aplicación y arrancar el sidecar cuando el puerto está libre.

## 2. Causa raíz técnica

La decisión de Tauri combina el proceso que escucha en el puerto 17493 con una
comprobación del contrato JSON de `/health`. Un test que solo comprobara si el
puerto está libre no distinguiría un backend Voicebox de un servicio ajeno.

## 3. Fix aplicado

`test_backend_detection.sh` reproduce el contrato de `main.rs` en tres casos:
backend activo, listener no-Voicebox y puerto libre. Usa un puerto alternativo
mediante `VOICEBOX_PORT` para no interrumpir una instalación activa.

## 4. Verificación

```bash
VOICEBOX_PORT=18493 bash scripts/test_backend_detection.sh --case B
VOICEBOX_PORT=18493 bash scripts/test_backend_detection.sh --case C
```

El caso A se ejecuta cuando existe un backend real en el puerto seleccionado.
El script no lanza ni mata la AppImage: valida el contrato de decisión y la
presencia del sidecar/build path.

## 5. Estado upstream

Pendiente de ejecución con Tauri/AppImage real. El harness es una prueba de
contrato, no reemplaza una prueba visual o de proceso de la aplicación.
