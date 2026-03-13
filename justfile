# Voicebox development commands
# Install: brew install just (or cargo install just)
# Usage: just --list

# Directories
backend_dir := "backend"
tauri_dir := "tauri"
app_dir := "app"
web_dir := "web"
venv := backend_dir / "venv"
venv_bin := venv / "bin"
python := venv_bin / "python"
pip := venv_bin / "pip"

# Detect best python for venv creation
system_python := `command -v python3.12 2>/dev/null || command -v python3.13 2>/dev/null || echo python3`

# ─── Setup ────────────────────────────────────────────────────────────

# Full project setup (python venv + JS deps + dev sidecar)
setup: setup-python setup-js
    @echo ""
    @echo "Setup complete! Run: just dev"

# Create venv and install Python dependencies
setup-python:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -d "{{ venv }}" ]; then
        echo "Creating Python virtual environment..."
        PY_MINOR=$({{ system_python }} -c "import sys; print(sys.version_info[1])")
        if [ "$PY_MINOR" -gt 13 ]; then
            echo "Warning: Python 3.$PY_MINOR detected. ML packages may not be compatible."
            echo "Recommended: brew install python@3.12"
        fi
        {{ system_python }} -m venv {{ venv }}
    fi
    echo "Installing Python dependencies..."
    {{ pip }} install --upgrade pip -q
    {{ pip }} install -r {{ backend_dir }}/requirements.txt
    # Chatterbox pins numpy<1.26 / torch==2.6 which break on Python 3.12+
    {{ pip }} install --no-deps chatterbox-tts
    # Apple Silicon: install MLX backend
    if [ "$(uname -m)" = "arm64" ] && [ "$(uname)" = "Darwin" ]; then
        echo "Detected Apple Silicon — installing MLX dependencies..."
        {{ pip }} install -r {{ backend_dir }}/requirements-mlx.txt
    fi
    {{ pip }} install git+https://github.com/QwenLM/Qwen3-TTS.git
    echo "Python environment ready."

# Install JavaScript dependencies
setup-js:
    bun install

# ─── Development ──────────────────────────────────────────────────────

# Start backend + frontend for development (two processes, one terminal)
dev: _ensure-venv _ensure-sidecar
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'kill 0' EXIT

    echo "Starting backend on http://localhost:17493 ..."
    {{ venv_bin }}/uvicorn backend.main:app --reload --port 17493 &
    sleep 2

    echo "Starting Tauri desktop app..."
    cd {{ tauri_dir }} && bun run tauri dev &

    wait

# Start backend only
dev-backend: _ensure-venv
    {{ venv_bin }}/uvicorn backend.main:app --reload --port 17493

# Start Tauri desktop app only (backend must be running separately)
dev-frontend: _ensure-sidecar
    cd {{ tauri_dir }} && bun run tauri dev

# Start backend + web app (no Tauri)
dev-web: _ensure-venv
    #!/usr/bin/env bash
    set -euo pipefail
    trap 'kill 0' EXIT
    {{ venv_bin }}/uvicorn backend.main:app --reload --port 17493 &
    sleep 2
    cd {{ web_dir }} && bun run dev &
    wait

# Kill all dev processes
kill:
    -pkill -f "uvicorn backend.main:app" 2>/dev/null || true
    -pkill -f "vite" 2>/dev/null || true
    @echo "Dev processes killed."

# ─── Build ────────────────────────────────────────────────────────────

# Build everything (server binary + desktop app)
build: build-server build-tauri

# Build Python server binary
build-server: _ensure-venv
    PATH="{{ venv_bin }}:$PATH" ./scripts/build-server.sh

# Build Tauri desktop app
build-tauri:
    cd {{ tauri_dir }} && bun run tauri build

# Build web app
build-web:
    cd {{ web_dir }} && bun run build

# ─── Code Quality ────────────────────────────────────────────────────

# Run all checks (lint + format + typecheck)
check:
    bun run check

# Lint with Biome
lint:
    bun run lint

# Format with Biome
format:
    bun run format

# Fix lint + format issues
fix:
    bun run check:fix

# ─── Database ─────────────────────────────────────────────────────────

# Initialize SQLite database
db-init: _ensure-venv
    cd {{ backend_dir }} && {{ python }} -c "from database import init_db; init_db()"

# Reset database (delete + reinit)
db-reset:
    rm -f {{ backend_dir }}/data/voicebox.db
    just db-init

# ─── Utilities ────────────────────────────────────────────────────────

# Generate TypeScript API client (backend must be running)
generate-api:
    ./scripts/generate-api.sh

# Open API docs in browser
docs:
    open http://localhost:17493/docs 2>/dev/null || xdg-open http://localhost:17493/docs

# Tail backend logs
logs:
    tail -f {{ backend_dir }}/logs/*.log 2>/dev/null || echo "No log files found"

# ─── Clean ────────────────────────────────────────────────────────────

# Clean build artifacts
clean:
    rm -rf {{ tauri_dir }}/src-tauri/target/release
    rm -rf {{ web_dir }}/dist
    rm -rf {{ app_dir }}/dist

# Clean Python venv and cache
clean-python:
    rm -rf {{ venv }}
    find {{ backend_dir }} -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Nuclear clean (everything including node_modules)
clean-all: clean clean-python
    rm -rf node_modules
    rm -rf {{ app_dir }}/node_modules
    rm -rf {{ tauri_dir }}/node_modules
    rm -rf {{ web_dir }}/node_modules
    cd {{ tauri_dir }}/src-tauri && cargo clean

# ─── Internal ─────────────────────────────────────────────────────────

# Ensure venv exists (prompt to run setup if not)
[private]
_ensure-venv:
    #!/usr/bin/env bash
    if [ ! -d "{{ venv }}" ]; then
        echo "Python venv not found. Run: just setup"
        exit 1
    fi

# Ensure Tauri dev sidecar placeholder exists
[private]
_ensure-sidecar:
    bun run setup:dev
