# Voicebox 中文启动指南

<p align="center">
  <img src=".github/assets/icon-dark.webp" alt="Voicebox" width="120" height="120" />
</p>

<h1 align="center">Voicebox</h1>

<p align="center">
  <strong>开源 AI 语音工作室</strong><br/>
  本地优先的语音克隆、语音生成、全局听写、AI 代理语音输出
</p>

---

## 📋 目录

- [环境要求](#环境要求)
- [快速开始（双端启动）](#快速开始双端启动)
- [后端启动（Conda 方式）](#后端启动conda-方式)
- [前端启动（Bun 方式）](#前端启动bun-方式)
- [桌面应用开发](#桌面应用开发)
- [Web 版本开发](#web-版本开发)
- [常见问题](#常见问题)

---

## 🔧 环境要求

在开始之前，请确保已安装以下工具：

| 工具 | 版本要求 | 说明 |
|------|----------|------|
| **Conda** (Miniconda/Anaconda) | 最新版 | Python 环境管理，推荐 Miniconda |
| **Python** | 3.12+ | 后端运行环境 |
| **Bun** | >= 1.0.0 | JavaScript/TypeScript 运行时与包管理器 |
| **Git** | 最新版 | 版本控制 |
| **Rust** | 最新稳定版 | 仅桌面应用（Tauri）需要 |

### 安装 Conda

推荐使用 Miniconda（轻量版）：

```bash
# Linux/macOS 下载安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 或访问官网下载：https://docs.conda.io/en/latest/miniconda.html
```

### 安装 Bun

```bash
# Linux/macOS 一键安装
curl -fsSL https://bun.sh/install | bash

# 重启终端或执行
source ~/.bashrc  # 或 source ~/.zshrc
```

验证安装：

```bash
conda --version
python --version
bun --version
```

---

## 🚀 快速开始（双端启动）

这是最推荐的开发方式：**Conda 管理后端 Python 环境 + Bun 管理前端依赖，双端并行启动**。

### 第一步：克隆项目

```bash
git clone https://github.com/jamiepine/voicebox.git
cd voicebox
```

### 第二步：配置后端（Conda 环境）

```bash
# 1. 创建并激活 conda 环境
conda create -n voicebox python=3.12 -y
conda activate voicebox

# 2. 升级 pip
pip install --upgrade pip

# 3. 安装 PyTorch（根据你的硬件选择）

# --- NVIDIA GPU (CUDA 12.8) ---
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# --- AMD GPU (ROCm 6.3) ---
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

# --- Apple Silicon (Mac M系列) ---
# pip install torch torchaudio

# --- CPU 版本（无GPU加速） ---
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. 安装后端依赖
pip install -r backend/requirements.txt

# 5. 安装特殊依赖（无依赖冲突安装）
pip install --no-deps chatterbox-tts
pip install --no-deps hume-tada

# 6. 安装 Qwen3-TTS
pip install git+https://github.com/QwenLM/Qwen3-TTS.git

# 7. （可选）Apple Silicon 安装 MLX 加速
if [[ "$(uname -m)" == "arm64" && "$(uname)" == "Darwin" ]]; then
    pip install -r backend/requirements-mlx.txt
    pip install --no-deps mlx-lm==0.31.1
    pip install --no-deps mlx-audio==0.4.1
fi

# 8. 安装开发工具
pip install pyinstaller ruff pytest pytest-asyncio
```

### 第三步：配置前端（Bun 依赖）

```bash
# 在项目根目录执行，安装所有工作区依赖
bun install
```

### 第四步：双端并行启动

打开**两个终端**窗口，分别启动后端和前端：

#### 终端 1：启动后端（Conda 环境）

```bash
conda activate voicebox
uvicorn backend.main:app --reload --port 17493
```

后端启动成功后，访问 http://127.0.0.1:17493/docs 可以看到 API 文档。

#### 终端 2：启动前端（Web 版本）

```bash
# 启动 Web 版本前端
cd web
bun run dev
```

前端启动后，访问 http://localhost:5173 即可使用。

---

## 🐍 后端启动（Conda 方式）

### 常用后端启动命令

```bash
# 确保已激活 conda 环境
conda activate voicebox

# 方式1：标准开发模式（自动重载）
uvicorn backend.main:app --reload --port 17493

# 方式2：指定主机地址（允许局域网访问）
uvicorn backend.main:app --reload --host 0.0.0.0 --port 17493

# 方式3：直接运行 main.py
python -m backend.main --host 127.0.0.1 --port 17493

# 方式4：自定义数据目录
python -m backend.main --data-dir ./data
```

### Conda 环境管理

```bash
# 创建环境
conda create -n voicebox python=3.12 -y

# 激活环境
conda activate voicebox

# 退出环境
conda deactivate

# 查看所有环境
conda env list

# 删除环境（谨慎使用）
conda env remove -n voicebox

# 导出环境配置
conda env export > environment.yml

# 从配置文件创建环境
conda env create -f environment.yml
```

### 后端代码检查与测试

```bash
conda activate voicebox

# 代码检查
ruff check backend/

# 代码格式化
ruff format backend/

# 自动修复 lint 问题
ruff check backend/ --fix

# 运行测试
python -m pytest backend/tests -v
```

---

## 🟨 前端启动（Bun 方式）

### Web 版本开发

```bash
# 在 web 目录下启动开发服务器
cd web
bun run dev

# 构建生产版本
bun run build

# 预览构建结果
bun run preview
```

### 桌面应用前端（配合 Tauri）

如果你要开发桌面应用，需要在 `tauri` 目录启动：

```bash
# 需要先确保后端已在 17493 端口运行
cd tauri
bun run tauri dev
```

### 营销网站（Landing Page）开发

```bash
cd landing
bun run dev
```

### 文档网站开发

```bash
cd docs
bun run dev
```

### 前端代码质量检查

```bash
# 在项目根目录执行

# Lint 检查
bun run lint

# TypeScript 类型检查
bun run typecheck

# 代码格式化
bun run format

# 自动修复所有问题
bun run check:fix
```

### Bun 常用命令

```bash
# 安装依赖
bun install

# 添加依赖
bun add <package-name>

# 添加开发依赖
bun add -d <package-name>

# 移除依赖
bun remove <package-name>

# 运行 package.json 中的脚本
bun run <script-name>
```

---

## 🖥️ 桌面应用开发

桌面应用使用 Tauri（Rust + Web 技术）开发，需要额外安装 Rust 工具链。

### 安装 Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.bashrc  # 或 source ~/.zshrc
```

### macOS 额外依赖

```bash
xcode-select --install
```

### 启动桌面应用开发模式

```bash
# 终端1：启动后端（如果未运行）
conda activate voicebox
uvicorn backend.main:app --reload --port 17493

# 终端2：启动 Tauri 桌面应用
bun run dev
```

或者使用 just 命令（如果已安装 just）：

```bash
# 需要先使用 venv 方式设置，或自行调整 justfile 使用 conda
just dev
```

### 构建桌面应用

```bash
bun run build
```

---

## 🌐 Web 版本开发

Web 版本可以直接在浏览器中运行，无需安装桌面应用。

```bash
# 终端1：启动后端
conda activate voicebox
uvicorn backend.main:app --reload --port 17493

# 终端2：启动 Web 前端
cd web
bun run dev
```

访问 http://localhost:5173 使用 Web 版本。

---

## 🔌 端口说明

| 服务 | 默认端口 | 说明 |
|------|----------|------|
| 后端 API | 17493 | FastAPI 服务，提供 REST API 和 MCP 服务 |
| Web 前端 | 5173 | Vite 开发服务器 |
| API 文档 | 17493/docs | Swagger UI 交互式文档 |

---

## ❓ 常见问题

### Q: 如何确认后端启动成功？

访问 http://127.0.0.1:17493/health，如果返回 JSON 状态信息则表示成功：

```bash
curl http://127.0.0.1:17493/health
```

### Q: 模型下载慢怎么办？

可以设置 HuggingFace 镜像源：

```bash
# Linux/macOS
export HF_ENDPOINT=https://hf-mirror.com

# Windows PowerShell
# $env:HF_ENDPOINT = "https://hf-mirror.com"
```

### Q: 首次启动需要下载模型吗？

是的，首次使用某个 TTS/STT 引擎时会自动从 HuggingFace 下载模型。下载的模型会缓存到本地，后续启动无需重新下载。

### Q: 如何切换 GPU/CPU 后端？

后端会自动检测硬件环境：
- macOS Apple Silicon → 自动使用 MLX (Metal)
- NVIDIA GPU → 使用 CUDA 加速
- AMD GPU → 使用 ROCm 加速
- 其他 → CPU 回退

可以通过环境变量强制指定：

```bash
# 强制使用 CPU
VOICEBOX_FORCE_CPU=1 uvicorn backend.main:app --reload --port 17493
```

### Q: Conda 环境和 venv 有什么区别？

| 特性 | Conda | venv |
|------|-------|------|
| Python 版本管理 | ✅ 可以安装不同 Python 版本 | ❌ 使用系统 Python |
| 非 Python 依赖 | ✅ 可以管理 CUDA、CuDNN 等 | ❌ 仅 Python 包 |
| 跨平台 | ✅ Windows/macOS/Linux 一致 | ✅ 跨平台 |
| 隔离级别 | 更彻底 | 轻量级 |

本指南推荐使用 Conda，因为它可以更好地管理 PyTorch、CUDA 等复杂依赖。

### Q: 如何完全重置开发环境？

```bash
# 1. 删除 Conda 环境
conda deactivate
conda env remove -n voicebox

# 2. 删除 node_modules
rm -rf node_modules app/node_modules tauri/node_modules web/node_modules landing/node_modules docs/node_modules

# 3. 清除 Python 缓存
find backend -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find backend -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null

# 4. 重新按照本文档步骤设置
```

---

## 📚 更多文档

- [官方文档](https://docs.voicebox.sh)（英文）
- [后端 README](backend/README.md)
- [贡献指南](CONTRIBUTING.md)
- [故障排除](docs/content/docs/overview/troubleshooting.mdx)（英文）

---

## 📝 开发命令速查表

| 操作 | 命令 |
|------|------|
| 激活后端环境 | `conda activate voicebox` |
| 启动后端 | `uvicorn backend.main:app --reload --port 17493` |
| 安装前端依赖 | `bun install` |
| 启动 Web 前端 | `cd web && bun run dev` |
| 启动桌面应用 | `bun run dev`（需先启动后端） |
| 后端 lint | `ruff check backend/` |
| 前端 lint | `bun run lint` |
| 后端测试 | `python -m pytest backend/tests -v` |
| 类型检查 | `bun run typecheck` |
| 构建 Web 版本 | `cd web && bun run build` |

---

<p align="center">
  <a href="https://voicebox.sh">voicebox.sh</a>
</p>
