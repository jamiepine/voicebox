# Contributing to Voicebox

Thank you for your interest in contributing to Voicebox! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

## Getting Started

### Prerequisites

- **[Bun](https://bun.sh)** - Fast JavaScript runtime and package manager
  ```bash
  curl -fsSL https://bun.sh/install | bash
  ```

- **[Python 3.11+](https://python.org)** - For backend development
  ```bash
  python --version  # Should be 3.11 or higher
  ```

- **[Rust](https://rustup.rs)** - For Tauri desktop app (installed automatically by Tauri CLI)
  ```bash
  rustc --version  # Check if installed
  ```
- **[Tauri Prerequisites](https://v2.tauri.app/start/prerequisites)** - Tauri-specific system dependencies (varies by OS).

- **Git** - Version control

### Development Setup

Install [just](https://github.com/casey/just) (`brew install just`, `cargo install just`, or `winget install Casey.Just`), then:

```bash
git clone https://github.com/YOUR_USERNAME/voicebox.git
cd voicebox

just setup   # creates venv, installs Python + JS deps
just dev     # starts backend + desktop app
```

`just setup` handles everything automatically, including:
- Creating a Python virtual environment
- Installing Python dependencies (with CUDA PyTorch on Windows if an NVIDIA GPU is detected)
- Installing MLX dependencies on Apple Silicon
- Installing JavaScript dependencies

`just dev` starts the backend and desktop app together. If a backend is already running (e.g. from `just dev-backend` in another terminal), it detects it and only starts the frontend.

Other useful commands:

```bash
just dev-web       # backend + web app (no Tauri/Rust build)
just dev-backend   # backend only
just dev-frontend  # Tauri app only (backend must be running)
just kill          # stop all dev processes
just clean-all     # nuke everything and start fresh
just --list        # see all available commands
```

> **Note:** In dev mode, the app connects to a manually-started Python server.
> The bundled server binary is only used in production builds.

#### Windows Notes

The justfile works natively on Windows via PowerShell. No WSL or Git Bash required. On Windows with an NVIDIA GPU, `just setup` automatically installs CUDA-enabled PyTorch for GPU acceleration.

### Model Downloads

Models are automatically downloaded from HuggingFace Hub on first use:
- **Whisper** (transcription): Auto-downloads on first transcription
- **Qwen3-TTS** (voice cloning): Auto-downloads on first generation (~2-4GB)

First-time usage will be slower due to model downloads, but subsequent runs will use cached models.

### Building

**Build production app:**

```bash
just build        # Build CPU server binary + Tauri installer
```

On Windows, to build with CUDA support for local testing:

```bash
just build-local  # Build CPU + CUDA server binaries + Tauri installer
```

This builds the CPU sidecar (bundled with the app), the CUDA binary (placed in `%APPDATA%/com.voicebox.app/backends/` for runtime GPU switching), and the installable Tauri app.

Creates platform-specific installers (`.dmg`, `.msi`, `.AppImage`) in `tauri/src-tauri/target/release/bundle/`.

**Individual build targets:**

```bash
just build-server       # CPU server binary only
just build-server-cuda  # CUDA server binary only (Windows)
just build-tauri        # Tauri desktop app only
just build-web          # Web app only
```

**Building with local Qwen3-TTS development version:**

If you're actively developing or modifying the Qwen3-TTS library, set the `QWEN_TTS_PATH` environment variable to point to your local clone:

```bash
export QWEN_TTS_PATH=~/path/to/your/Qwen3-TTS
just build-server
```

This makes PyInstaller use your local qwen-tts version instead of the pip-installed package.

### Generate OpenAPI Client

After starting the backend server:
```bash
./scripts/generate-api.sh
```
This downloads the OpenAPI schema and generates the TypeScript client in `app/src/lib/api/`

### Convert Assets to Web Formats

To optimize images and videos for the web, run:
```bash
bun run convert:assets
```

This script:
- Converts PNG → WebP (better compression, same quality)
- Converts MOV → WebM (VP9 codec, smaller file size)
- Processes files in `landing/public/` and `docs/public/`
- **Deletes original files** after successful conversion

**Requirements:** Install `webp` and `ffmpeg`:
```bash
brew install webp ffmpeg
```

> **Note:** Run this before committing new images or videos to keep the repository size small.

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clean, readable code
- Follow existing code style
- Add comments for complex logic
- Update documentation as needed

### 3. Test Your Changes

- Test manually in the app
- Ensure backend API endpoints work
- Check for TypeScript/Python errors
- Verify UI components render correctly

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git commit -m "Add feature: voice profile export"
git commit -m "Fix: audio playback stops after 30 seconds"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear description of changes
- Screenshots (for UI changes)
- Reference to related issues

## Code Style

### TypeScript/React

- Use TypeScript strict mode
- Follow React best practices
- Use functional components with hooks
- Prefer named exports
- Format with Biome (runs automatically)

```typescript
// Good
export function ProfileCard({ profile }: { profile: Profile }) {
  return <div>{profile.name}</div>;
}

// Avoid
export const ProfileCard = (props) => { ... }
```

### Python

- Follow PEP 8 style guide
- Use type hints
- Use async/await for I/O operations
- Format with Black (if configured)

```python
# Good
async def create_profile(name: str, language: str) -> Profile:
    """Create a new voice profile."""
    ...

# Avoid
def create_profile(name, language):
    ...
```

### Rust

- Follow Rust conventions
- Use meaningful variable names
- Handle errors explicitly
- Format with `rustfmt`

## Project Structure

```
voicebox/
├── app/              # Shared React frontend
│   └── src/
│       ├── components/   # UI components
│       ├── lib/          # Utilities and API client
│       └── hooks/        # React hooks
├── backend/          # Python FastAPI server
│   ├── main.py       # API routes
│   ├── tts.py        # Voice synthesis
│   └── ...
├── tauri/            # Desktop app wrapper
│   └── src-tauri/    # Rust backend
└── scripts/          # Build scripts
```

## Areas for Contribution

### 🐛 Bug Fixes

- Check existing issues for bugs to fix
- Test your fix thoroughly
- Add tests if possible

### ✨ New Features

- Check the roadmap in README.md and the engineering status in [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) before proposing work — it lists prioritized tasks (Tier 1 → 3), known architectural bottlenecks, and candidate TTS engines already under evaluation (including why some have been backlogged)
- Discuss major features in an issue first
- Keep features focused and well-scoped

### 📚 Documentation

- Improve README clarity
- Add code comments
- Write API documentation
- Create tutorials or guides

### 🎨 UI/UX Improvements

- Improve accessibility
- Enhance visual design
- Optimize performance
- Add animations/transitions

### 🔧 Infrastructure

- Improve build process
- Add CI/CD improvements
- Optimize bundle size
- Add testing infrastructure

## API Development

When adding new API endpoints:

1. **Add route in `backend/main.py`**
2. **Create Pydantic models in `backend/models.py`**
3. **Implement business logic in appropriate module**
4. **Update OpenAPI schema** (automatic with FastAPI)
5. **Regenerate TypeScript client:**
   ```bash
   bun run generate:api
   ```
6. **Update `backend/README.md`** with endpoint documentation

## Testing

Currently, testing is primarily manual. When adding tests:

- **Backend**: Use pytest for Python tests
- **Frontend**: Use Vitest for React component tests
- **E2E**: Use Playwright for end-to-end tests (future)

## Pull Request Process

1. **Update documentation** if needed
2. **Ensure code follows style guidelines**
3. **Test your changes thoroughly**
4. **Update CHANGELOG.md** with your changes
5. **Request review** from maintainers

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Changes tested
- [ ] No breaking changes (or documented)
- [ ] CHANGELOG.md updated

## Release Process

Releases are managed by maintainers:

1. **Bump version using bumpversion:**
   ```bash
   # Install bumpversion (if not already installed)
   pip install bumpversion
   
   # Bump patch version (0.1.0 -> 0.1.1)
   bumpversion patch
   
   # Or bump minor version (0.1.0 -> 0.2.0)
   bumpversion minor
   
   # Or bump major version (0.1.0 -> 1.0.0)
   bumpversion major
   ```
   
   This automatically:
   - Updates version numbers in all files (`tauri.conf.json`, `Cargo.toml`, all `package.json` files, `backend/main.py`)
   - Creates a git commit with the version bump
   - Creates a git tag (e.g., `v0.1.1`, `v0.2.0`)

2. **Update CHANGELOG.md** with release notes

3. **Push commits and tags:**
   ```bash
   git push
   git push --tags
   ```

4. **GitHub Actions builds and releases** automatically when tags are pushed

## Troubleshooting

See [docs/content/docs/overview/troubleshooting.mdx](docs/content/docs/overview/troubleshooting.mdx) for common issues and solutions.

**Quick fixes:**

- **Backend won't start:** Check Python version (3.11+), ensure venv is activated, install dependencies
- **Tauri build fails:** Ensure Rust is installed, clean build with `cd tauri/src-tauri && cargo clean`
- **OpenAPI client generation fails:** Ensure backend is running, check `curl http://localhost:17493/openapi.json`

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues and discussions
- Review the codebase to understand patterns
- See [docs/content/docs/overview/troubleshooting.mdx](docs/content/docs/overview/troubleshooting.mdx) for common issues

## Additional Resources

- [README.md](README.md) - Project overview
- [backend/README.md](backend/README.md) - API documentation
- [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) - Living engineering roadmap: architecture, shipped vs in-flight work, prioritized open issues, candidate TTS engines under evaluation, architectural bottlenecks. Keep this updated when you ship significant features, close or backlog a model integration, or identify new bottlenecks.
- [docs/AUTOUPDATER_QUICKSTART.md](docs/AUTOUPDATER_QUICKSTART.md) - Auto-updater setup
- [SECURITY.md](SECURITY.md) - Security policy
- [CHANGELOG.md](CHANGELOG.md) - Version history

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Voicebox! 🎉
