# Contribuir a Voicebox

[English](CONTRIBUTING.md) · **Español**

¡Gracias por tu interés en contribuir a Voicebox! Este documento ofrece pautas e instrucciones para contribuir.

## Código de conducta

- Sé respetuoso e inclusivo
- Da la bienvenida a los recién llegados y ayúdales a aprender
- Céntrate en el feedback constructivo
- Respeta los distintos puntos de vista y experiencias

## Primeros pasos

### Requisitos previos

- **[Bun](https://bun.sh)** - Runtime y gestor de paquetes de JavaScript rápido
  ```bash
  curl -fsSL https://bun.sh/install | bash
  ```

- **[Python 3.11+](https://python.org)** - Para el desarrollo del backend
  ```bash
  python --version  # Debe ser 3.11 o superior
  ```

- **[Rust](https://rustup.rs)** - Para la app de escritorio Tauri (lo instala automáticamente la CLI de Tauri)
  ```bash
  rustc --version  # Comprueba si está instalado
  ```
- **[Requisitos previos de Tauri](https://v2.tauri.app/start/prerequisites)** - Dependencias del sistema específicas de Tauri (varían según el SO).

- **Git** - Control de versiones

### Configuración del entorno de desarrollo

Instala [just](https://github.com/casey/just) (`brew install just`, `cargo install just` o `winget install Casey.Just`), y luego:

```bash
git clone https://github.com/YOUR_USERNAME/voicebox.git
cd voicebox

just setup   # crea el venv, instala dependencias de Python + JS
just dev     # arranca el backend + la app de escritorio
```

`just setup` se encarga de todo automáticamente, incluyendo:
- Crear un entorno virtual de Python
- Instalar las dependencias de Python (con PyTorch CUDA en Windows si se detecta una GPU NVIDIA)
- Instalar las dependencias de MLX en Apple Silicon
- Instalar las dependencias de JavaScript

`just dev` arranca el backend y la app de escritorio juntos. Si ya hay un backend en marcha (p. ej. desde `just dev-backend` en otra terminal), lo detecta y solo arranca el frontend.

Otros comandos útiles:

```bash
just dev-web       # backend + app web (sin compilación de Tauri/Rust)
just dev-backend   # solo backend
just dev-frontend  # solo app Tauri (el backend debe estar en marcha)
just kill          # detiene todos los procesos de desarrollo
just clean-all     # borra todo y empieza de cero
just --list        # ve todos los comandos disponibles
```

> **Nota:** En modo desarrollo, la app se conecta a un servidor Python iniciado manualmente.
> El binario del servidor incluido solo se usa en las compilaciones de producción.

#### Notas para Windows

El justfile funciona de forma nativa en Windows vía PowerShell. No hace falta WSL ni Git Bash. En Windows con una GPU NVIDIA, `just setup` instala automáticamente PyTorch con CUDA para la aceleración por GPU.

### Descargas de modelos

Los modelos se descargan automáticamente desde HuggingFace Hub en el primer uso:
- **Whisper** (transcripción): se descarga solo en la primera transcripción
- **Qwen3-TTS** (clonación de voz): se descarga solo en la primera generación (~2-4 GB)

El primer uso será más lento por las descargas de modelos, pero las ejecuciones posteriores usarán los modelos en caché.

### Compilación

**Compilar la app de producción:**

```bash
just build        # Compila el binario del servidor CPU + el instalador de Tauri
```

En Windows, para compilar con compatibilidad CUDA para pruebas locales:

```bash
just build-local  # Compila los binarios del servidor CPU + CUDA + el instalador de Tauri
```

Esto compila el sidecar de CPU (incluido con la app), el binario CUDA (ubicado en `%APPDATA%/com.voicebox.app/backends/` para el cambio de GPU en tiempo de ejecución) y la app Tauri instalable.

Crea instaladores específicos de cada plataforma (`.dmg`, `.msi`, `.AppImage`) en `tauri/src-tauri/target/release/bundle/`.

**Objetivos de compilación individuales:**

```bash
just build-server       # solo el binario del servidor CPU
just build-server-cuda  # solo el binario del servidor CUDA (Windows)
just build-tauri        # solo la app de escritorio Tauri
just build-web          # solo la app web
```

**Compilar con una versión de desarrollo local de Qwen3-TTS:**

Si estás desarrollando o modificando activamente la biblioteca Qwen3-TTS, define la variable de entorno `QWEN_TTS_PATH` apuntando a tu clon local:

```bash
export QWEN_TTS_PATH=~/path/to/your/Qwen3-TTS
just build-server
```

Esto hace que PyInstaller use tu versión local de qwen-tts en lugar del paquete instalado con pip.

### Generar el cliente OpenAPI

Tras arrancar el servidor backend:
```bash
./scripts/generate-api.sh
```
Esto descarga el esquema OpenAPI y genera el cliente TypeScript en `app/src/lib/api/`

### Convertir recursos a formatos web

Para optimizar imágenes y vídeos para la web, ejecuta:
```bash
bun run convert:assets
```

Este script:
- Convierte PNG → WebP (mejor compresión, misma calidad)
- Convierte MOV → WebM (códec VP9, menor tamaño de archivo)
- Procesa los archivos en `landing/public/` y `docs/public/`
- **Elimina los archivos originales** tras una conversión correcta

**Requisitos:** Instala `webp` y `ffmpeg`:
```bash
brew install webp ffmpeg
```

> **Nota:** Ejecuta esto antes de hacer commit de imágenes o vídeos nuevos para mantener el tamaño del repositorio pequeño.

## Flujo de trabajo de desarrollo

### 1. Crea una rama

```bash
git checkout -b feature/your-feature-name
# o
git checkout -b fix/your-bug-fix
```

### 2. Haz tus cambios

- Escribe código limpio y legible
- Sigue el estilo de código existente
- Añade comentarios para la lógica compleja
- Actualiza la documentación según haga falta

### 3. Prueba tus cambios

- Pruébalos manualmente en la app
- Asegúrate de que los endpoints de la API del backend funcionan
- Comprueba que no hay errores de TypeScript/Python
- Verifica que los componentes de la UI se renderizan correctamente

### 4. Haz commit de tus cambios

Escribe mensajes de commit claros y descriptivos:

```bash
git commit -m "Add feature: voice profile export"
git commit -m "Fix: audio playback stops after 30 seconds"
```

### 5. Haz push y crea un Pull Request

```bash
git push origin feature/your-feature-name
```

Luego crea un pull request en GitHub con:
- Una descripción clara de los cambios
- Capturas de pantalla (para cambios de UI)
- Referencia a los issues relacionados

## Estilo de código

### TypeScript/React

- Usa el modo estricto de TypeScript
- Sigue las buenas prácticas de React
- Usa componentes funcionales con hooks
- Prefiere las exportaciones nombradas
- Formatea con Biome (se ejecuta automáticamente)

```typescript
// Bien
export function ProfileCard({ profile }: { profile: Profile }) {
  return <div>{profile.name}</div>;
}

// Evita
export const ProfileCard = (props) => { ... }
```

### Python

- Sigue la guía de estilo PEP 8
- Usa anotaciones de tipo
- Usa async/await para las operaciones de E/S
- Formatea con Black (si está configurado)

```python
# Bien
async def create_profile(name: str, language: str) -> Profile:
    """Create a new voice profile."""
    ...

# Evita
def create_profile(name, language):
    ...
```

### Rust

- Sigue las convenciones de Rust
- Usa nombres de variable significativos
- Maneja los errores de forma explícita
- Formatea con `rustfmt`

## Estructura del proyecto

```
voicebox/
├── app/              # Frontend React compartido
│   └── src/
│       ├── components/   # Componentes de UI
│       ├── lib/          # Utilidades y cliente de la API
│       └── hooks/        # Hooks de React
├── backend/          # Servidor Python FastAPI
│   ├── main.py       # Rutas de la API
│   ├── tts.py        # Síntesis de voz
│   └── ...
├── tauri/            # Envoltorio de la app de escritorio
│   └── src-tauri/    # Backend en Rust
└── scripts/          # Scripts de compilación
```

## Áreas para contribuir

### 🐛 Corrección de errores

- Revisa los issues existentes en busca de errores que corregir
- Prueba tu corrección a fondo
- Añade tests si es posible

### ✨ Funciones nuevas

- Revisa la hoja de ruta en README.md y el estado de ingeniería en [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) antes de proponer trabajo — enumera tareas priorizadas (Tier 1 → 3), cuellos de botella arquitectónicos conocidos y motores TTS candidatos ya en evaluación (incluido por qué algunos se han aplazado)
- Discute las funciones grandes primero en un issue
- Mantén las funciones enfocadas y bien acotadas

### 📚 Documentación

- Mejora la claridad del README
- Añade comentarios al código
- Escribe documentación de la API
- Crea tutoriales o guías

### 🎨 Mejoras de UI/UX

- Mejora la accesibilidad
- Mejora el diseño visual
- Optimiza el rendimiento
- Añade animaciones/transiciones

### 🔧 Infraestructura

- Mejora el proceso de compilación
- Añade mejoras de CI/CD
- Optimiza el tamaño del bundle
- Añade infraestructura de tests

## Desarrollo de la API

Al añadir nuevos endpoints de la API:

1. **Añade la ruta en `backend/main.py`**
2. **Crea los modelos Pydantic en `backend/models.py`**
3. **Implementa la lógica de negocio en el módulo apropiado**
4. **Actualiza el esquema OpenAPI** (automático con FastAPI)
5. **Regenera el cliente TypeScript:**
   ```bash
   bun run generate:api
   ```
6. **Actualiza `backend/README.md`** con la documentación del endpoint

## Tests

Actualmente, las pruebas son principalmente manuales. Al añadir tests:

- **Backend**: usa pytest para los tests de Python
- **Frontend**: usa Vitest para los tests de componentes de React
- **E2E**: usa Playwright para los tests de extremo a extremo (futuro)

## Proceso de Pull Request

1. **Actualiza la documentación** si hace falta
2. **Asegúrate de que el código sigue las pautas de estilo**
3. **Prueba tus cambios a fondo**
4. **Actualiza CHANGELOG.md** con tus cambios
5. **Solicita revisión** a los mantenedores

### Checklist del PR

- [ ] El código sigue las pautas de estilo
- [ ] Documentación actualizada
- [ ] Cambios probados
- [ ] Sin cambios disruptivos (o documentados)
- [ ] CHANGELOG.md actualizado

## Proceso de release

Los releases los gestionan los mantenedores:

1. **Sube la versión con bumpversion:**
   ```bash
   # Instala bumpversion (si aún no está instalado)
   pip install bumpversion
   
   # Sube la versión de parche (0.1.0 -> 0.1.1)
   bumpversion patch
   
   # O sube la versión menor (0.1.0 -> 0.2.0)
   bumpversion minor
   
   # O sube la versión mayor (0.1.0 -> 1.0.0)
   bumpversion major
   ```
   
   Esto automáticamente:
   - Actualiza los números de versión en todos los archivos (`tauri.conf.json`, `Cargo.toml`, todos los `package.json`, `backend/main.py`)
   - Crea un commit de git con la subida de versión
   - Crea una etiqueta de git (p. ej., `v0.1.1`, `v0.2.0`)

2. **Actualiza CHANGELOG.md** con las notas del release

3. **Haz push de los commits y las etiquetas:**
   ```bash
   git push
   git push --tags
   ```

4. **GitHub Actions compila y publica** automáticamente cuando se suben las etiquetas

## Solución de problemas

Consulta [docs/content/docs/overview/troubleshooting.mdx](docs/content/docs/overview/troubleshooting.mdx) para problemas comunes y sus soluciones.

**Arreglos rápidos:**

- **El backend no arranca:** Comprueba la versión de Python (3.11+), asegúrate de que el venv está activado, instala las dependencias
- **La compilación de Tauri falla:** Asegúrate de que Rust está instalado, limpia la compilación con `cd tauri/src-tauri && cargo clean`
- **La generación del cliente OpenAPI falla:** Asegúrate de que el backend está en marcha, comprueba `curl http://localhost:17493/openapi.json`

## ¿Preguntas?

- Abre un issue para errores o solicitudes de funciones
- Revisa los issues y discusiones existentes
- Revisa el código para entender los patrones
- Consulta [docs/content/docs/overview/troubleshooting.mdx](docs/content/docs/overview/troubleshooting.mdx) para problemas comunes

## Recursos adicionales

- [README.md](README.md) - Visión general del proyecto
- [backend/README.md](backend/README.md) - Documentación de la API
- [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) - Hoja de ruta de ingeniería viva: arquitectura, trabajo publicado vs en curso, issues abiertos priorizados, motores TTS candidatos en evaluación, cuellos de botella arquitectónicos. Mantenlo actualizado cuando publiques funciones importantes, cierres o aplaces una integración de modelo, o identifiques nuevos cuellos de botella.
- [docs/content/docs/developer/autoupdater.mdx](docs/content/docs/developer/autoupdater.mdx) - Configuración del actualizador automático
- [SECURITY.md](SECURITY.md) - Política de seguridad
- [CHANGELOG.md](CHANGELOG.md) - Historial de versiones

## Licencia

Al contribuir, aceptas que tus contribuciones se licencien bajo la Licencia MIT.

---

¡Gracias por contribuir a Voicebox! 🎉
