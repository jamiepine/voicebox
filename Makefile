# Voicebox — top-level Makefile
# Single entry point: install everything and run the app from a fresh clone.
#
# For the full task runner (build-server, release, Windows CUDA, etc.) use `just`.
# This Makefile is the newcomer-friendly wrapper; just is the power-user tool.

# ── Platform detection ────────────────────────────────────────────────────────

OS   := $(shell uname -s)
ARCH := $(shell uname -m)

# ── Paths ─────────────────────────────────────────────────────────────────────

BACKEND_DIR   := backend
APP_DIR       := app
TAURI_DIR     := tauri
VENV          := $(BACKEND_DIR)/venv
VENV_BIN      := $(VENV)/bin
PYTHON        := $(VENV_BIN)/python
PIP           := $(VENV_BIN)/pip

# Pick the best available Python 3.12 binary for venv creation
SYSTEM_PYTHON := $(shell command -v python3.12 2>/dev/null || command -v python3.13 2>/dev/null || echo python3)

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Install ───────────────────────────────────────────────────────────────────

.PHONY: install
install: install-system install-deps ## Full install from scratch (brew deps + Python venv + JS packages)

.PHONY: install-system
install-system:  ## Install system dependencies (Mac: brew, Linux: apt/dnf)
ifeq ($(OS),Darwin)
	@which brew > /dev/null || (echo "Install Homebrew first: https://brew.sh" && exit 1)
	brew bundle --file=Brewfile
else ifeq ($(OS),Linux)
	@$(MAKE) install-system-linux
else
	@echo "Unsupported OS: $(OS). Install manually: python3.12, bun, rustup, ffmpeg"
endif

.PHONY: install-system-linux
install-system-linux:  ## Install system dependencies on Linux
	@echo "Detecting Linux package manager..."
	@if command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv python3.12-dev ffmpeg webp build-essential; \
	elif command -v dnf >/dev/null 2>&1; then \
		sudo dnf install -y python3.12 python3.12-devel ffmpeg libwebp-tools gcc; \
	elif command -v pacman >/dev/null 2>&1; then \
		sudo pacman -Sy --noconfirm python ffmpeg libwebp base-devel; \
	else \
		echo "Unknown package manager. Install manually: python3.12, ffmpeg, webp, build tools"; \
		exit 1; \
	fi
	@# Install bun if not present
	@command -v bun >/dev/null 2>&1 || curl -fsSL https://bun.sh/install | bash
	@# Install rustup if not present
	@command -v rustup >/dev/null 2>&1 || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
	@# Ensure cargo is on PATH (needed right after a fresh rustup install)
	@export PATH="$$HOME/.cargo/bin:$$PATH"; \
	command -v cargo >/dev/null 2>&1 || { echo "cargo not found after rustup install"; exit 1; }
	@# Install just
	@command -v just >/dev/null 2>&1 || PATH="$$HOME/.cargo/bin:$$PATH" cargo install just
	@# Install pre-commit
	@command -v pre-commit >/dev/null 2>&1 || pip3 install pre-commit

.PHONY: install-deps
install-deps: install-python install-node ## Install Python venv and JS packages

.PHONY: install-python
install-python: ## Create Python 3.12 venv and install backend dependencies
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating Python virtual environment with $(SYSTEM_PYTHON)..."; \
		$(SYSTEM_PYTHON) -m venv $(VENV); \
	fi
	$(PIP) install --upgrade pip -q
	$(PIP) install -r $(BACKEND_DIR)/requirements.txt
	@# Chatterbox pins numpy<1.26 / torch==2.6 which conflict on Python 3.12+
	$(PIP) install --no-deps chatterbox-tts
	@# HumeAI TADA pins torch>=2.7,<2.8 which conflicts with our torch>=2.1
	$(PIP) install --no-deps hume-tada
ifeq ($(OS),Darwin)
ifeq ($(ARCH),arm64)
	@echo "Apple Silicon detected — installing MLX backend..."
	$(PIP) install -r $(BACKEND_DIR)/requirements-mlx.txt
	$(PIP) install --no-deps mlx-audio==0.4.1
endif
else
	@echo "Linux detected — installing CUDA/CPU backend deps if present"
	@test -f $(BACKEND_DIR)/requirements-linux.txt && $(PIP) install -r $(BACKEND_DIR)/requirements-linux.txt || true
endif
	$(PIP) install git+https://github.com/QwenLM/Qwen3-TTS.git
	$(PIP) install pyinstaller ruff pytest pytest-asyncio -q
	@echo "Python environment ready."

.PHONY: install-node
install-node: ## Install frontend dependencies (bun workspaces)
	bun install
	bun run setup:dev

.PHONY: install-rust
install-rust: ## Install Rust stable toolchain via rustup
	@which rustup > /dev/null || (echo "Installing rustup..." && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y)
	rustup toolchain install stable
	@echo "Rust toolchain ready: $$(rustc --version)"

# ── Dev ───────────────────────────────────────────────────────────────────────

.PHONY: dev
dev: ## Start backend + Tauri frontend in parallel (requires tmux or two terminals)
	@echo "Starting backend and frontend in parallel..."
	@echo "  Backend:  http://localhost:17493"
	@echo "  API docs: http://localhost:17493/docs"
	$(MAKE) -j2 dev-backend dev-frontend

.PHONY: dev-backend
dev-backend: ## Run the Python FastAPI backend only
	@[ -d "$(VENV)" ] || (echo "Run 'make install-python' first." && exit 1)
	cd $(BACKEND_DIR) && ../$(VENV_BIN)/python main.py

.PHONY: dev-frontend
dev-frontend: ## Run the Tauri desktop app (Vite + Rust)
	@which bun > /dev/null || (echo "bun not found — run 'make install-system'" && exit 1)
	cd $(TAURI_DIR) && bun run tauri dev

.PHONY: dev-web
dev-web: ## Run the web app (no Tauri/Rust build needed)
	@which bun > /dev/null || (echo "bun not found — run 'make install-system'" && exit 1)
	cd web && bun run dev

# ── Build ─────────────────────────────────────────────────────────────────────

.PHONY: build
build: ## Build production Tauri app (server binary + installer)
	./scripts/build-server.sh
	cd $(TAURI_DIR) && bun run tauri build

.PHONY: build-frontend
build-frontend: ## Build Tauri app only (skips server binary)
	cd $(TAURI_DIR) && bun run tauri build

.PHONY: build-web
build-web: ## Build web app
	cd web && bun run build

# ── Test ──────────────────────────────────────────────────────────────────────

.PHONY: test
test: test-backend test-frontend ## Run all tests (backend + frontend)

.PHONY: test-backend
test-backend: ## Run all backend pytest tests
	@[ -d "$(VENV)" ] || (echo "Run 'make install-python' first." && exit 1)
	cd $(BACKEND_DIR) && ../$(PYTHON) -m pytest tests/ -x -q

.PHONY: test-unit
test-unit: ## Run backend unit tests only (fast, no I/O)
	@[ -d "$(VENV)" ] || (echo "Run 'make install-python' first." && exit 1)
	cd $(BACKEND_DIR) && ../$(PYTHON) -m pytest tests/unit -x -q

.PHONY: test-integration
test-integration: ## Run backend integration tests (real SQLite, no external network)
	@[ -d "$(VENV)" ] || (echo "Run 'make install-python' first." && exit 1)
	cd $(BACKEND_DIR) && ../$(PYTHON) -m pytest tests/integration -x -q

.PHONY: test-frontend
test-frontend: ## Run frontend Vitest tests
	@which bun > /dev/null || (echo "bun not found — run 'make install-system'" && exit 1)
	cd $(APP_DIR) && bun run test:run

# ── Lint / Format ─────────────────────────────────────────────────────────────

.PHONY: lint
lint: ## Run all linters (ruff + biome)
	@[ -d "$(VENV)" ] || (echo "Run 'make install-python' first." && exit 1)
	cd $(BACKEND_DIR) && ../$(VENV_BIN)/ruff check .
	cd $(BACKEND_DIR) && ../$(VENV_BIN)/ruff format --check .
	bun run lint

.PHONY: format
format: ## Auto-fix all formatting (ruff + biome)
	@[ -d "$(VENV)" ] || (echo "Run 'make install-python' first." && exit 1)
	cd $(BACKEND_DIR) && ../$(VENV_BIN)/ruff check --fix .
	cd $(BACKEND_DIR) && ../$(VENV_BIN)/ruff format .
	bun run check:fix

.PHONY: typecheck
typecheck: ## Run TypeScript type checks
	bun run typecheck

# ── Pre-commit ────────────────────────────────────────────────────────────────

.PHONY: pre-commit-install
pre-commit-install: ## Install pre-commit hooks into .git/hooks
	@[ -d "$(VENV)" ] || (echo "Run 'make install-python' first." && exit 1)
	$(VENV_BIN)/pre-commit install

.PHONY: pre-commit-run
pre-commit-run: ## Run all pre-commit hooks against staged files
	@[ -d "$(VENV)" ] || (echo "Run 'make install-python' first." && exit 1)
	$(VENV_BIN)/pre-commit run

# ── Utilities ─────────────────────────────────────────────────────────────────

.PHONY: generate-api
generate-api: ## Regenerate TypeScript API client (backend must be running)
	./scripts/generate-api.sh

.PHONY: db-reset
db-reset: ## Delete and reinitialise the SQLite database
	rm -f $(BACKEND_DIR)/data/voicebox.db
	$(PYTHON) -c "from backend.database import init_db; init_db()"

# ── Clean ─────────────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove build artifacts (Rust target release, JS dist)
	rm -rf $(TAURI_DIR)/src-tauri/target/release
	rm -rf web/dist
	rm -rf $(APP_DIR)/dist

.PHONY: clean-python
clean-python: ## Remove Python venv and __pycache__ directories
	rm -rf $(VENV)
	find $(BACKEND_DIR) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

.PHONY: clean-all
clean-all: clean clean-python ## Nuclear clean — removes venv, node_modules, and Rust target
	rm -rf node_modules
	rm -rf $(APP_DIR)/node_modules
	rm -rf $(TAURI_DIR)/node_modules
	rm -rf web/node_modules
	cd $(TAURI_DIR)/src-tauri && cargo clean
