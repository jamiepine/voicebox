# Política de seguridad

[English](SECURITY.md) · **Español**

## Versiones compatibles

Publicamos parches para vulnerabilidades de seguridad. Qué versiones son elegibles para recibir dichos parches depende de la clasificación CVSS v3.0:

| Versión | Compatible         |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reportar una vulnerabilidad

Si descubres una vulnerabilidad de seguridad, repórtala de forma responsable:

1. **No** abras un issue público en GitHub
2. Envía los detalles de seguridad por correo a: [security@voicebox.sh](mailto:security@voicebox.sh)
3. Incluye:
   - Descripción de la vulnerabilidad
   - Pasos para reproducirla
   - Impacto potencial
   - Solución sugerida (si la hay)

Nosotros:
- Confirmaremos la recepción en un plazo de 48 horas
- Daremos un calendario para abordar el problema
- Te mantendremos informado del progreso
- Te daremos crédito en el aviso de seguridad (si lo deseas)

## Buenas prácticas de seguridad

### Para usuarios

- **Mantén Voicebox actualizado** - Las actualizaciones incluyen parches de seguridad
- **Verifica las descargas** - Descarga solo de los releases oficiales
- **Procesamiento local** - Los datos de voz se quedan en tu equipo
- **Seguridad de red** - Usa HTTPS al conectarte a servidores remotos

### Para desarrolladores

- **Dependencias** - Mantén todas las dependencias al día
- **Revisión de código** - Todos los PR requieren revisión antes de fusionarse
- **Secretos** - Nunca hagas commit de claves de API ni de firma
- **Firma** - Todos los releases están firmados criptográficamente

## Consideraciones de seguridad conocidas

### Procesamiento local

Voicebox procesa todo el audio localmente de forma predeterminada. Tus datos de voz nunca salen de tu equipo a menos que actives explícitamente el modo de servidor remoto.

### Modo de servidor remoto

Al conectarte a un servidor remoto:
- Asegúrate de que el servidor está en una red de confianza
- Usa HTTPS para las conexiones remotas
- Verifica la identidad del servidor antes de conectarte

### Actualizaciones automáticas

- Las actualizaciones están firmadas criptográficamente
- La verificación de la firma ocurre antes de la instalación
- Solo se permiten endpoints HTTPS

### Servidor Python

El servidor Python integrado:
- Se ejecuta localmente de forma predeterminada (solo localhost)
- Se puede configurar para acceso remoto
- Usa las prácticas de seguridad estándar de FastAPI

## Calendario de divulgación

- **Día 0**: Vulnerabilidad reportada
- **Día 1-2**: Evaluación inicial y confirmación
- **Día 3-7**: Investigación y desarrollo de la corrección
- **Día 8-14**: Pruebas y preparación del release
- **Día 15+**: Divulgación pública (si procede)

El calendario puede variar según la gravedad y la complejidad.

## Actualizaciones de seguridad

Las actualizaciones de seguridad se:
- Publicarán como versiones de parche (p. ej., 0.3.2)
- Documentarán en CHANGELOG.md
- Anunciarán vía releases de GitHub
- Entregarán automáticamente vía el actualizador automático

---

¡Gracias por ayudar a mantener Voicebox seguro! 🔒
