#!/usr/bin/env bash
set -euo pipefail

# Voicebox Linux setup script
# Installs the backend server with CUDA support on x86_64 Linux

INSTALL_DIR="/opt/voicebox"
DATA_DIR="/var/lib/voicebox"
VENV_DIR="$INSTALL_DIR/venv"
SERVICE_USER="voicebox"
PYTHON_MIN="3.10"
CUDA_MIN="11.8"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[x]${NC} $*" >&2; }
die()   { error "$*"; exit 1; }

usage() {
    cat <<EOF
Voicebox Linux Setup

Usage: $0 <command>

Commands:
  check       Check system requirements (GPU, Python, etc.)
  install     Install voicebox server to $INSTALL_DIR
  service     Install and enable systemd service
  uninstall   Remove voicebox installation

Options:
  --no-cuda       Skip CUDA, use CPU only
  --install-dir   Custom install dir (default: $INSTALL_DIR)
  --data-dir      Custom data dir (default: $DATA_DIR)
  --port          Server port (default: 17493)

EOF
    exit 0
}

# --- Checks ---

check_root() {
    if [[ $EUID -ne 0 ]]; then
        die "This script must be run as root (try: sudo $0 $*)"
    fi
}

check_arch() {
    local arch
    arch=$(uname -m)
    if [[ "$arch" != "x86_64" ]]; then
        die "Unsupported architecture: $arch (need x86_64)"
    fi
    info "Architecture: $arch"
}

check_python() {
    local py=""
    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" &>/dev/null; then
            py="$candidate"
            break
        fi
    done

    if [[ -z "$py" ]]; then
        die "Python 3.10+ not found. Install with: apt install python3 python3-venv python3-pip"
    fi

    local ver
    ver=$($py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
        info "Python: $py ($ver)"
    else
        die "Python $ver is too old (need $PYTHON_MIN+)"
    fi

    # Check venv module
    if ! $py -c "import venv" 2>/dev/null; then
        die "Python venv module missing. Install with: apt install python3-venv"
    fi

    PYTHON_BIN="$py"
}

check_nvidia() {
    if [[ "${NO_CUDA:-0}" == "1" ]]; then
        warn "CUDA skipped (--no-cuda)"
        return
    fi

    info "Checking NVIDIA GPU..."

    # Check for nvidia device nodes
    if [[ ! -e /dev/nvidia0 ]]; then
        warn "/dev/nvidia0 not found — NVIDIA driver may not be loaded"
        warn "Try: nvidia-smi  or  modprobe nvidia"
    else
        info "Device: /dev/nvidia0 exists"
        # Check permissions
        local perms
        perms=$(stat -c '%a' /dev/nvidia0 2>/dev/null || echo "???")
        local group
        group=$(stat -c '%G' /dev/nvidia0 2>/dev/null || echo "???")
        info "  /dev/nvidia0 permissions: $perms (group: $group)"

        if [[ ! -r /dev/nvidia0 ]] || [[ ! -w /dev/nvidia0 ]]; then
            warn "  Current user cannot access /dev/nvidia0"
            warn "  Fix: usermod -aG $group $SERVICE_USER"
        fi
    fi

    if [[ ! -e /dev/nvidiactl ]]; then
        warn "/dev/nvidiactl not found"
    fi

    if [[ ! -e /dev/nvidia-uvm ]]; then
        warn "/dev/nvidia-uvm not found (needed for CUDA)"
        warn "Fix: modprobe nvidia-uvm"
    fi

    # nvidia-smi
    if ! command -v nvidia-smi &>/dev/null; then
        warn "nvidia-smi not found — NVIDIA driver not installed?"
        warn "Install: apt install nvidia-driver-535 (or newer)"
        return
    fi

    local driver_ver gpu_name gpu_mem
    driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1)
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)

    info "GPU: $gpu_name ($gpu_mem)"
    info "Driver: $driver_ver"

    # Check CUDA toolkit
    if command -v nvcc &>/dev/null; then
        local cuda_ver
        cuda_ver=$(nvcc --version | grep -oP 'release \K[\d.]+')
        info "CUDA toolkit: $cuda_ver"
    else
        warn "nvcc not found — CUDA toolkit not installed (PyTorch bundles its own, this is OK)"
    fi
}

check_system() {
    info "=== System Check ==="
    check_arch
    check_python
    check_nvidia

    # Disk space
    local avail
    avail=$(df -BG --output=avail / | tail -1 | tr -d ' G')
    if (( avail < 15 )); then
        warn "Low disk space: ${avail}G free (models need ~10G)"
    else
        info "Disk: ${avail}G available"
    fi

    # RAM
    local ram_gb
    ram_gb=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
    if (( ram_gb < 8 )); then
        warn "Low RAM: ${ram_gb}G (recommend 16G+)"
    else
        info "RAM: ${ram_gb}G"
    fi

    echo
    info "=== Check complete ==="
}

# --- Install ---

do_install() {
    check_root
    check_arch
    check_python

    local src_dir
    src_dir="$(cd "$(dirname "$0")" && pwd)"

    info "Installing voicebox to $INSTALL_DIR..."

    # Create service user
    if ! id "$SERVICE_USER" &>/dev/null; then
        info "Creating user: $SERVICE_USER"
        useradd --system --home-dir "$INSTALL_DIR" --shell /usr/sbin/nologin "$SERVICE_USER"
    fi

    # Add to video group for GPU access
    if getent group video &>/dev/null; then
        usermod -aG video "$SERVICE_USER"
        info "Added $SERVICE_USER to video group (GPU access)"
    fi
    if getent group render &>/dev/null; then
        usermod -aG render "$SERVICE_USER"
        info "Added $SERVICE_USER to render group (GPU access)"
    fi

    # Create directories
    mkdir -p "$INSTALL_DIR" "$DATA_DIR"

    # Copy backend source
    info "Copying backend source..."
    rsync -a --delete \
        --exclude='venv/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='.claude/' \
        "$src_dir/" "$INSTALL_DIR/backend/"

    # Create venv
    info "Creating Python venv..."
    $PYTHON_BIN -m venv "$VENV_DIR"

    # Install PyTorch with CUDA
    info "Installing dependencies (this may take a while)..."
    if [[ "${NO_CUDA:-0}" == "1" ]]; then
        "$VENV_DIR/bin/pip" install --upgrade pip
        "$VENV_DIR/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu
    else
        "$VENV_DIR/bin/pip" install --upgrade pip
        "$VENV_DIR/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu121
    fi

    "$VENV_DIR/bin/pip" install -r "$INSTALL_DIR/backend/requirements.txt"

    # Copy CLI
    cp "$src_dir/../voicebox" "$INSTALL_DIR/voicebox" 2>/dev/null || true

    # Create launcher
    cat > /usr/local/bin/voicebox <<LAUNCHER
#!/bin/bash
source "$VENV_DIR/bin/activate"
exec python "$INSTALL_DIR/backend/cli.py" "\$@"
LAUNCHER
    chmod +x /usr/local/bin/voicebox

    # Fix ownership
    chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR" "$DATA_DIR"

    echo
    info "=== Install complete ==="
    info "Server dir:  $INSTALL_DIR"
    info "Data dir:    $DATA_DIR"
    info "CLI:         /usr/local/bin/voicebox"
    echo
    info "Next steps:"
    info "  voicebox server -d --data-dir $DATA_DIR"
    info "  voicebox import <voice.zip>"
    info "  voicebox say -v 'Will' -t 'Hello from Linux'"
    echo
    info "Or install as a systemd service:"
    info "  sudo $0 service"
}

# --- Systemd ---

do_service() {
    check_root

    local port="${PORT:-17493}"

    cat > /etc/systemd/system/voicebox.service <<SERVICE
[Unit]
Description=Voicebox TTS Server
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$VENV_DIR/bin/python -m backend.main --host 0.0.0.0 --port $port
Environment=HOME=$INSTALL_DIR
Environment=HF_HOME=$DATA_DIR/huggingface
Restart=on-failure
RestartSec=5

# GPU access
SupplementaryGroups=video render

# Hardening
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=$DATA_DIR $INSTALL_DIR
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
SERVICE

    systemctl daemon-reload
    systemctl enable voicebox
    systemctl start voicebox

    info "Systemd service installed and started"
    info "  Status:  systemctl status voicebox"
    info "  Logs:    journalctl -u voicebox -f"
    info "  Stop:    systemctl stop voicebox"
}

# --- Uninstall ---

do_uninstall() {
    check_root

    warn "This will remove voicebox from $INSTALL_DIR"
    read -rp "Continue? [y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || exit 0

    # Stop service
    if systemctl is-active voicebox &>/dev/null; then
        systemctl stop voicebox
    fi
    if [[ -f /etc/systemd/system/voicebox.service ]]; then
        systemctl disable voicebox 2>/dev/null || true
        rm -f /etc/systemd/system/voicebox.service
        systemctl daemon-reload
    fi

    rm -rf "$INSTALL_DIR"
    rm -f /usr/local/bin/voicebox

    info "Removed $INSTALL_DIR and /usr/local/bin/voicebox"
    warn "Data dir preserved at $DATA_DIR (delete manually if unwanted)"

    # Don't remove user if data dir still exists
    if id "$SERVICE_USER" &>/dev/null; then
        warn "User '$SERVICE_USER' preserved (remove with: userdel $SERVICE_USER)"
    fi
}

# --- Parse args ---

NO_CUDA=0
PORT=17493
COMMAND=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        check|install|service|uninstall)
            COMMAND="$1" ;;
        --no-cuda)
            NO_CUDA=1 ;;
        --install-dir)
            INSTALL_DIR="$2"; VENV_DIR="$INSTALL_DIR/venv"; shift ;;
        --data-dir)
            DATA_DIR="$2"; shift ;;
        --port)
            PORT="$2"; shift ;;
        -h|--help)
            usage ;;
        *)
            die "Unknown argument: $1" ;;
    esac
    shift
done

case "$COMMAND" in
    check)     check_system ;;
    install)   do_install ;;
    service)   do_service ;;
    uninstall) do_uninstall ;;
    *)         usage ;;
esac
