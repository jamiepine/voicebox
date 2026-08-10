# Runbook local de dictado

Este flujo es local y no forma parte de la instalación upstream: usa el
script de usuario `~/.local/bin/voicebox-dictate.sh` y herramientas del
entorno gráfico Linux.

## Preparación

- Backend Voicebox saludable en `127.0.0.1:17493`.
- Script de dictado instalado en `~/.local/bin/voicebox-dictate.sh`.
- Herramientas disponibles según la sesión: `arecord` o `pw-record`,
  `curl`, `wl-copy`, `ydotool` y `notify-send`.

## Harness

```bash
VOICEBOX_DICTATE_SCRIPT="$HOME/.local/bin/voicebox-dictate.sh" \
  bash scripts/test_dictate_e2e.sh
```

El harness usa stubs y un `HOME` temporal. No modifica la caché de Voicebox,
el portapapeles real ni el script instalado. Valida la cadena de grabación,
transcripción, copia y pegado, además de fallos de cada etapa.

La prueba no puede validar permisos reales de Wayland, disponibilidad de
`ydotoold`, audio físico ni interacción visual de una sesión de escritorio.
