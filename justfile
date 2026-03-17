# Voicebox development commands
# Install: brew install just (or cargo install just)
# Usage: just --list

# Directories
backend_dir := "backend"
tauri_dir := "tauri"
app_dir := "app"
web_dir := "web"
venv := backend_dir / "venv"

# Platform-aware paths
venv_bin := if os() == "windows" { venv / "Scripts" } else { venv / "bin" }
python := if os() == "windows" { venv_bin / "python.exe" } else { venv_bin / "python" }
pip := if os() == "windows" { venv_bin / "pip.exe" } else { venv_bin / "pip" }

# Shell selection: use powershell on Windows, bash elsewhere
set windows-shell := ["powershell", "-NoProfile", "-Command"]

# Detect best python for venv creation (platform-aware)
system_python := if os() == "windows" { "python" } else { `command -v python3.12 2>/dev/null || command -v python3.13 2>/dev/null || echo python3` }

# ─── Setup ────────────────────────────────────────────────────────────

# Full project setup (python venv + JS deps + dev sidecar)
setup: setup-python setup-js
    @echo ""
    @echo "Setup complete! Run: just dev"

# Create venv and install Python dependencies
[unix]
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
    # HumeAI TADA pins torch>=2.7,<2.8 which conflicts with our torch>=2.1
    {{ pip }} install --no-deps hume-tada
    # Apple Silicon: install MLX backend
    if [ "$(uname -m)" = "arm64" ] && [ "$(uname)" = "Darwin" ]; then
        echo "Detected Apple Silicon — installing MLX dependencies..."
        {{ pip }} install -r {{ backend_dir }}/requirements-mlx.txt
    fi
    {{ pip }} install git+https://github.com/QwenLM/Qwen3-TTS.git
    {{ pip }} install pyinstaller ruff pytest pytest-asyncio -q
    echo "Python environment ready."

[windows]
setup-python:
    if (-not (Test-Path "{{ venv }}")) { \
        Write-Host "Creating Python virtual environment..."; \
        $pyMinor = & {{ system_python }} -c "import sys; print(sys.version_info[1])"; \
        if ([int]$pyMinor -gt 13) { \
            Write-Host "Warning: Python 3.$pyMinor detected. ML packages may not be compatible."; \
        }; \
        & {{ system_python }} -m venv {{ venv }}; \
    }
    Write-Host "Installing Python dependencies..."
    & "{{ python }}" -m pip install --upgrade pip -q
    $hasNvidia = $null -ne (Get-WmiObject Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' })
    if ($hasNvidia) { \
        Write-Host "NVIDIA GPU detected — installing PyTorch with CUDA support..."; \
        & "{{ pip }}" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126; \
    }
    & "{{ pip }}" install -r {{ backend_dir }}/requirements.txt
    & "{{ pip }}" install --no-deps chatterbox-tts
    & "{{ pip }}" install --no-deps hume-tada
    & "{{ pip }}" install git+https://github.com/QwenLM/Qwen3-TTS.git
    & "{{ pip }}" install pyinstaller ruff pytest pytest-asyncio -q
    Write-Host "Python environment ready."

# Install JavaScript dependencies
setup-js:
    bun install

# ─── Development ──────────────────────────────────────────────────────

# Start backend (if not already running) + frontend for development
[unix]
dev: _ensure-venv _ensure-sidecar
    #!/usr/bin/env bash
    set -euo pipefail

    backend_pid=""
    if curl -sf http://127.0.0.1:17493/health > /dev/null 2>&1; then
        echo "Backend already running on http://localhost:17493"
    else
        echo "Starting backend on http://localhost:17493 ..."
        {{ venv_bin }}/uvicorn backend.main:app --reload --port 17493 &
        backend_pid=$!
        sleep 2
    fi

    trap '[ -n "$backend_pid" ] && kill "$backend_pid" 2>/dev/null; wait' EXIT

    echo "Starting Tauri desktop app..."
    cd {{ tauri_dir }} && bun run tauri dev

[windows]
dev: _ensure-venv _ensure-sidecar
    $backendJob = $null; \
    try { $null = Invoke-WebRequest -Uri "http://127.0.0.1:17493/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop; Write-Host "Backend already running on http://localhost:17493" } catch { \
        Write-Host "Starting backend on http://localhost:17493 ..."; \
        $backendJob = Start-Process -PassThru -NoNewWindow -FilePath "{{ python }}" -ArgumentList "-m","uvicorn","backend.main:app","--reload","--port","17493"; \
        Start-Sleep -Seconds 2; \
    }; \
    Write-Host "Starting Tauri desktop app..."; \
    try { Set-Location "{{ tauri_dir }}"; bun run tauri dev } finally { if ($backendJob) { taskkill /PID $backendJob.Id /T /F 2>$null | Out-Null } }

# Start backend only
[unix]
dev-backend: _ensure-venv
    {{ venv_bin }}/uvicorn backend.main:app --reload --port 17493

[windows]
dev-backend: _ensure-venv
    & "{{ python }}" -m uvicorn backend.main:app --reload --port 17493

# Start Tauri desktop app only (backend must be running separately)
[unix]
dev-frontend: _ensure-sidecar
    cd {{ tauri_dir }} && bun run tauri dev

[windows]
dev-frontend: _ensure-sidecar
    Set-Location "{{ tauri_dir }}"; bun run tauri dev

# Start backend (if not already running) + web app (no Tauri)
[unix]
dev-web: _ensure-venv
    #!/usr/bin/env bash
    set -euo pipefail

    backend_pid=""
    if curl -sf http://127.0.0.1:17493/health > /dev/null 2>&1; then
        echo "Backend already running on http://localhost:17493"
    else
        echo "Starting backend on http://localhost:17493 ..."
        {{ venv_bin }}/uvicorn backend.main:app --reload --port 17493 &
        backend_pid=$!
        sleep 2
    fi

    trap '[ -n "$backend_pid" ] && kill "$backend_pid" 2>/dev/null; wait' EXIT

    cd {{ web_dir }} && bun run dev

[windows]
dev-web: _ensure-venv
    $backendJob = $null; \
    try { $null = Invoke-WebRequest -Uri "http://127.0.0.1:17493/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop; Write-Host "Backend already running on http://localhost:17493" } catch { \
        Write-Host "Starting backend on http://localhost:17493 ..."; \
        $backendJob = Start-Process -PassThru -NoNewWindow -FilePath "{{ python }}" -ArgumentList "-m","uvicorn","backend.main:app","--reload","--port","17493"; \
        Start-Sleep -Seconds 2; \
    }; \
    Write-Host "Starting web app..."; \
    try { Set-Location "{{ web_dir }}"; bun run dev } finally { if ($backendJob) { taskkill /PID $backendJob.Id /T /F 2>$null | Out-Null } }

# Kill all dev processes
[unix]
kill:
    -pkill -f "uvicorn backend.main:app" 2>/dev/null || true
    -pkill -f "vite" 2>/dev/null || true
    @echo "Dev processes killed."

[windows]
kill:
    Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*uvicorn*backend.main*' -or $_.CommandLine -like '*vite*' } | Stop-Process -Force -ErrorAction SilentlyContinue
    Write-Host "Dev processes killed."

# ─── Build ────────────────────────────────────────────────────────────

# Build everything (server binary + desktop app)
build: build-server build-tauri

# Build Python server binary (CPU)
[unix]
build-server: _ensure-venv
    PATH="{{ venv_bin }}:$PATH" ./scripts/build-server.sh

[windows]
build-server: _ensure-venv
    $ErrorActionPreference = "Stop"; \
    $env:PATH = "{{ venv_bin }};$env:PATH"; \
    & "{{ python }}" backend/build_binary.py; \
    if ($LASTEXITCODE -ne 0) { throw "build_binary.py failed with exit code $LASTEXITCODE" }; \
    $triple = (rustc --print host-tuple); \
    New-Item -ItemType Directory -Path "{{ tauri_dir }}/src-tauri/binaries" -Force | Out-Null; \
    Copy-Item "backend/dist/voicebox-server.exe" "{{ tauri_dir }}/src-tauri/binaries/voicebox-server-$triple.exe" -Force; \
    Write-Host "Copied sidecar: voicebox-server-$triple.exe"

# Build CUDA server binary and place in app data dir for local testing
[windows]
build-server-cuda: _ensure-venv
    $ErrorActionPreference = "Stop"; \
    $env:PATH = "{{ venv_bin }};$env:PATH"; \
    & "{{ python }}" backend/build_binary.py --cuda; \
    if ($LASTEXITCODE -ne 0) { throw "build_binary.py --cuda failed with exit code $LASTEXITCODE" }; \
    $dest = "$env:APPDATA/sh.voicebox.app/backends/cuda"; \
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }; \
    New-Item -ItemType Directory -Path $dest -Force | Out-Null; \
    Copy-Item "backend/dist/voicebox-server-cuda/*" $dest -Recurse -Force; \
    Write-Host "Copied CUDA backend to $dest"

# Build everything locally: CPU server + CUDA server + installable Tauri app
[windows]
build-local: build-server build-server-cuda build-tauri

# Build Tauri desktop app
[unix]
build-tauri:
    cd {{ tauri_dir }} && bun run tauri build

[windows]
build-tauri:
    Set-Location "{{ tauri_dir }}"; bun run tauri build

# Build web app
[unix]
build-web:
    cd {{ web_dir }} && bun run build

[windows]
build-web:
    Set-Location "{{ web_dir }}"; bun run build

# ─── Code Quality ────────────────────────────────────────────────────

# Run all checks (JS + Python lint + format)
check: check-js check-python

# JS/TS: lint + format + typecheck (Biome)
check-js:
    bun run check

# Python: lint + format check (ruff)
check-python: _ensure-venv
    {{ venv_bin }}/ruff check {{ backend_dir }}
    {{ venv_bin }}/ruff format --check {{ backend_dir }}

# Lint with Biome (JS) + ruff (Python)
lint: _ensure-venv
    bun run lint
    {{ venv_bin }}/ruff check {{ backend_dir }}

# Format with Biome (JS) + ruff (Python)
format: _ensure-venv
    bun run format
    {{ venv_bin }}/ruff format {{ backend_dir }}

# Fix lint + format issues (JS + Python)
fix: _ensure-venv
    bun run check:fix
    {{ venv_bin }}/ruff check {{ backend_dir }} --fix
    {{ venv_bin }}/ruff format {{ backend_dir }}

# Python lint only
lint-python: _ensure-venv
    {{ venv_bin }}/ruff check {{ backend_dir }}

# Python format only
format-python: _ensure-venv
    {{ venv_bin }}/ruff format {{ backend_dir }}

# Python auto-fix lint issues
fix-python: _ensure-venv
    {{ venv_bin }}/ruff check {{ backend_dir }} --fix
    {{ venv_bin }}/ruff format {{ backend_dir }}

# Run Python tests
test: _ensure-venv
    {{ venv_bin }}/python -m pytest {{ backend_dir }}/tests -v

# ─── Database ─────────────────────────────────────────────────────────

# Initialize SQLite database
[unix]
db-init: _ensure-venv
    {{ python }} -c "from backend.database import init_db; init_db()"

[windows]
db-init: _ensure-venv
    & "{{ python }}" -c "from backend.database import init_db; init_db()"

# Reset database (delete + reinit)
[unix]
db-reset:
    rm -f {{ backend_dir }}/data/voicebox.db
    just db-init

[windows]
db-reset:
    if (Test-Path "{{ backend_dir }}/data/voicebox.db") { Remove-Item -Force "{{ backend_dir }}/data/voicebox.db" }
    just db-init

# ─── Utilities ────────────────────────────────────────────────────────

# Generate TypeScript API client (backend must be running)
[unix]
generate-api:
    ./scripts/generate-api.sh

[windows]
generate-api:
    bash scripts/generate-api.sh

# Open API docs in browser
[unix]
docs:
    open http://localhost:17493/docs 2>/dev/null || xdg-open http://localhost:17493/docs

[windows]
docs:
    Start-Process "http://localhost:17493/docs"

# Tail backend logs
[unix]
logs:
    tail -f {{ backend_dir }}/logs/*.log 2>/dev/null || echo "No log files found"

[windows]
logs:
    Get-ChildItem {{ backend_dir }}/logs/*.log -ErrorAction SilentlyContinue | ForEach-Object { Get-Content $_.FullName -Tail 50 -Wait } ; if (-not $?) { Write-Host "No log files found" }

# ─── Clean ────────────────────────────────────────────────────────────

# Clean build artifacts
[unix]
clean:
    rm -rf {{ tauri_dir }}/src-tauri/target/release
    rm -rf {{ web_dir }}/dist
    rm -rf {{ app_dir }}/dist

[windows]
clean:
    if (Test-Path "{{ tauri_dir }}/src-tauri/target/release") { Remove-Item -Recurse -Force "{{ tauri_dir }}/src-tauri/target/release" }
    if (Test-Path "{{ web_dir }}/dist") { Remove-Item -Recurse -Force "{{ web_dir }}/dist" }
    if (Test-Path "{{ app_dir }}/dist") { Remove-Item -Recurse -Force "{{ app_dir }}/dist" }

# Clean Python venv and cache
[unix]
clean-python:
    rm -rf {{ venv }}
    find {{ backend_dir }} -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

[windows]
clean-python:
    if (Test-Path "{{ venv }}") { Remove-Item -Recurse -Force "{{ venv }}" }
    Get-ChildItem -Path "{{ backend_dir }}" -Directory -Recurse -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force

# Nuclear clean (everything including node_modules)
[unix]
clean-all: clean clean-python
    rm -rf node_modules
    rm -rf {{ app_dir }}/node_modules
    rm -rf {{ tauri_dir }}/node_modules
    rm -rf {{ web_dir }}/node_modules
    cd {{ tauri_dir }}/src-tauri && cargo clean

[windows]
clean-all: clean clean-python
    if (Test-Path "node_modules") { Remove-Item -Recurse -Force "node_modules" }
    if (Test-Path "{{ app_dir }}/node_modules") { Remove-Item -Recurse -Force "{{ app_dir }}/node_modules" }
    if (Test-Path "{{ tauri_dir }}/node_modules") { Remove-Item -Recurse -Force "{{ tauri_dir }}/node_modules" }
    if (Test-Path "{{ web_dir }}/node_modules") { Remove-Item -Recurse -Force "{{ web_dir }}/node_modules" }
    Push-Location "{{ tauri_dir }}/src-tauri"; cargo clean; Pop-Location

# ─── Internal ─────────────────────────────────────────────────────────

# Ensure venv exists (prompt to run setup if not)
[private, unix]
_ensure-venv:
    #!/usr/bin/env bash
    if [ ! -d "{{ venv }}" ]; then
        echo "Python venv not found. Run: just setup"
        exit 1
    fi

[private, windows]
_ensure-venv:
    if (-not (Test-Path "{{ venv }}")) { Write-Host "Python venv not found. Run: just setup"; exit 1 }

# Ensure Tauri dev sidecar placeholder exists
[private]
_ensure-sidecar:
    bun run setup:dev
