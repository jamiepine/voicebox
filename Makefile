# Voicebox development commands
# This is a Makefile equivalent of the justfile, using uv for Python management
# Usage: make help
#
# Prerequisites: uv (https://astral.sh/uv), Bun, Rust (for Tauri)
# Install uv:  curl -LsSf https://astral.sh/uv/install.sh | sh

SHELL := /bin/bash

# Directories
backend_dir := backend
tauri_dir := tauri
app_dir := app
web_dir := web

# uv-managed virtual environment (.venv in project root)
# uv run automatically detects and uses .venv — no manual activation needed
# Force uv pip to use .venv instead of the active conda/shell environment
export VIRTUAL_ENV := $(CURDIR)/.venv

# Default ARGS for test-models
ARGS ?=

# Default target
.DEFAULT_GOAL := help

# ─── Help ─────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo "Voicebox Makefile 命令列表（uv 版）"
	@echo ""
	@echo "环境搭建:"
	@echo "  make setup         完整项目搭建（uv sync + JS 依赖 + 开发 sidecar）"
	@echo "  make setup-python  通过 uv sync 安装所有 Python 依赖（含 dev 工具）"
	@echo "  make setup-js      安装 JavaScript 依赖"
	@echo ""
	@echo "开发:"
	@echo "  make dev           启动后端 + 前端用于开发"
	@echo "  make dev-backend   仅启动后端"
	@echo "  make dev-frontend  仅启动 Tauri 桌面应用"
	@echo "  make dev-web       启动后端 + Web 应用（无 Tauri）"
	@echo "  make kill          终止所有开发进程"
	@echo ""
	@echo "构建:"
	@echo "  make build         构建全部（服务器二进制 + 桌面应用）"
	@echo "  make build-server  构建 Python 服务器二进制（CPU 版）"
	@echo "  make build-tauri   构建 Tauri 桌面应用"
	@echo "  make build-web     构建 Web 应用"
	@echo ""
	@echo "代码质量:"
	@echo "  make check         运行所有检查（JS + Python 代码检查 + 格式化）"
	@echo "  make check-js      JS/TS：代码检查 + 格式化 + 类型检查（Biome）"
	@echo "  make check-python  Python：代码检查 + 格式化检查（ruff）"
	@echo "  make lint          代码检查（Biome JS + ruff Python）"
	@echo "  make format        格式化代码（Biome JS + ruff Python）"
	@echo "  make fix           自动修复代码检查 + 格式化问题（JS + Python）"
	@echo ""
	@echo "测试:"
	@echo "  make test          运行 Python 测试"
	@echo "  make test-models   端到端测试：用每个 TTS 模型生成语音"
	@echo ""
	@echo "数据库:"
	@echo "  make db-init       初始化 SQLite 数据库"
	@echo "  make db-reset      重置数据库（删除后重新初始化）"
	@echo ""
	@echo "实用工具:"
	@echo "  make generate-api  生成 TypeScript API 客户端"
	@echo "  make docs          在浏览器中打开 API 文档"
	@echo "  make logs          跟踪查看后端日志"
	@echo ""
	@echo "清理:"
	@echo "  make clean         清理构建产物"
	@echo "  make clean-python  清理 uv 虚拟环境和缓存"
	@echo "  make clean-all     彻底清理（包括 node_modules 等全部依赖）"

# ─── Setup ────────────────────────────────────────────────────────────

.PHONY: setup setup-python setup-js
setup: setup-python setup-js
	@echo ""
	@echo "环境搭建完成！运行: make dev"

setup-python:
	@set -euo pipefail; \
	echo "正在使用 uv sync 安装所有 Python 依赖..."; \
	uv sync; \
	if [ "$$(uname)" = "Linux" ]; then \
		torch_index=""; \
		if [ -e /proc/driver/nvidia/version ] || [ -d /sys/module/nvidia ]; then \
			echo "检测到 NVIDIA GPU —— 安装 CUDA PyTorch..."; \
			torch_index="https://download.pytorch.org/whl/cu128"; \
		elif [ -e /dev/kfd ]; then \
			if [ -n "$${VOICEBOX_ROCM_VERSION:-}" ]; then \
				rocm_ver="$$VOICEBOX_ROCM_VERSION"; \
			elif lspci 2>/dev/null | grep -qi "Navi 4"; then \
				rocm_ver=7.2; \
			else \
				rocm_ver=6.3; \
			fi; \
			echo "检测到 AMD GPU —— 安装 ROCm PyTorch (rocm$$rocm_ver)..."; \
			torch_index="https://download.pytorch.org/whl/rocm$$rocm_ver"; \
		fi; \
		if [ -n "$$torch_index" ]; then \
			uv pip install torch torchaudio --index-url "$$torch_index"; \
		fi; \
	fi; \
	echo "Python 环境准备就绪。"

setup-js:
	bun install

# ─── Development ──────────────────────────────────────────────────────

.PHONY: dev dev-backend dev-frontend dev-web kill
dev: _ensure-venv _ensure-sidecar
	@set -euo pipefail; \
	backend_pid=""; \
	if curl -sf http://127.0.0.1:17493/health > /dev/null 2>&1; then \
		echo "后端已在 http://localhost:17493 运行"; \
	else \
		echo "正在启动后端 http://localhost:17493 ..."; \
		uv run uvicorn backend.main:app --reload --port 17493 & \
		backend_pid=$$!; \
		sleep 2; \
	fi; \
	trap '[ -n "$$backend_pid" ] && kill "$$backend_pid" 2>/dev/null; wait' EXIT; \
	echo "正在启动 Tauri 桌面应用..."; \
	cd $(tauri_dir) && bun run tauri dev

dev-backend: _ensure-venv
	uv run uvicorn backend.main:app --reload --port 17493

dev-frontend: _ensure-sidecar
	cd $(tauri_dir) && bun run tauri dev

dev-web: _ensure-venv
	@set -euo pipefail; \
	backend_pid=""; \
	if curl -sf http://127.0.0.1:17493/health > /dev/null 2>&1; then \
		echo "后端已在 http://localhost:17493 运行"; \
	else \
		echo "正在启动后端 http://localhost:17493 ..."; \
		uv run uvicorn backend.main:app --reload --port 17493 & \
		backend_pid=$$!; \
		sleep 2; \
	fi; \
	trap '[ -n "$$backend_pid" ] && kill "$$backend_pid" 2>/dev/null; wait' EXIT; \
	cd $(web_dir) && bun run dev

kill:
	-pkill -f "uvicorn backend.main:app" 2>/dev/null || true
	-pkill -f "vite" 2>/dev/null || true
	@echo "开发进程已终止。"

# ─── Build ────────────────────────────────────────────────────────────

.PHONY: build build-server build-tauri build-web
build: build-server build-tauri

build-server: _ensure-venv
	uv run ./scripts/build-server.sh

build-tauri:
	cd $(tauri_dir) && bun run tauri build

build-web:
	cd $(web_dir) && bun run build

# ─── Code Quality ────────────────────────────────────────────────────

.PHONY: check check-js check-python lint format fix lint-python format-python fix-python
check: check-js check-python

check-js:
	bun run check

check-python: _ensure-venv
	uv run ruff check $(backend_dir)
	uv run ruff format --check $(backend_dir)

lint: _ensure-venv
	bun run lint
	uv run ruff check $(backend_dir)

format: _ensure-venv
	bun run format
	uv run ruff format $(backend_dir)

fix: _ensure-venv
	bun run check:fix
	uv run ruff check $(backend_dir) --fix
	uv run ruff format $(backend_dir)

lint-python: _ensure-venv
	uv run ruff check $(backend_dir)

format-python: _ensure-venv
	uv run ruff format $(backend_dir)

fix-python: _ensure-venv
	uv run ruff check $(backend_dir) --fix
	uv run ruff format $(backend_dir)

# ─── Test ─────────────────────────────────────────────────────────────

.PHONY: test test-models
test: _ensure-venv
	uv run python -m pytest $(backend_dir)/tests -v

test-models: _ensure-venv
	uv run python $(backend_dir)/tests/test_all_models_e2e.py $(ARGS)

# ─── Database ─────────────────────────────────────────────────────────

.PHONY: db-init db-reset
db-init: _ensure-venv
	uv run python -c "from backend.database import init_db; init_db()"

db-reset:
	rm -f $(backend_dir)/data/voicebox.db
	$(MAKE) db-init

# ─── Utilities ────────────────────────────────────────────────────────

.PHONY: generate-api docs logs
generate-api:
	./scripts/generate-api.sh

docs:
	open http://localhost:17493/docs 2>/dev/null || xdg-open http://localhost:17493/docs

logs:
	tail -f $(backend_dir)/logs/*.log 2>/dev/null || echo "No log files found"

# ─── Clean ────────────────────────────────────────────────────────────

.PHONY: clean clean-python clean-all
clean:
	rm -rf $(tauri_dir)/src-tauri/target/release
	rm -rf $(web_dir)/dist
	rm -rf $(app_dir)/dist

clean-python:
	rm -rf .venv
	find $(backend_dir) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

clean-all: clean clean-python
	rm -rf node_modules
	rm -rf $(app_dir)/node_modules
	rm -rf $(tauri_dir)/node_modules
	rm -rf $(web_dir)/node_modules
	cd $(tauri_dir)/src-tauri && cargo clean

# ─── Internal ─────────────────────────────────────────────────────────

# Ensure uv venv exists (prompt to run setup if not)
.PHONY: _ensure-venv
_ensure-venv:
	@if ! command -v uv > /dev/null 2>&1; then \
		echo "错误：未安装 uv。请先安装：curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	fi
	@if [ ! -d ".venv" ]; then \
		echo "错误：未找到 Python 虚拟环境。请运行: make setup"; \
		exit 1; \
	fi

# Ensure Tauri dev sidecar placeholder exists
.PHONY: _ensure-sidecar
_ensure-sidecar:
	bun run setup:dev
