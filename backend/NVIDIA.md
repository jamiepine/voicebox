# NVIDIA GPU Setup for Voicebox (Ubuntu/Debian)

## 1. Install NVIDIA Driver

```bash
# Ubuntu
sudo apt update
sudo apt install -y nvidia-driver-535
sudo reboot

# Debian
sudo apt update
sudo apt install -y linux-headers-$(uname -r)
sudo apt install -y nvidia-driver
sudo reboot
```

Verify after reboot:

```bash
nvidia-smi
```

You should see your GPU name, driver version, and CUDA version.

## 2. Verify Device Nodes

```bash
ls -la /dev/nvidia*
```

You need at minimum:
- `/dev/nvidia0` (GPU device)
- `/dev/nvidiactl` (control device)
- `/dev/nvidia-uvm` (CUDA unified memory)

If `/dev/nvidia-uvm` is missing:

```bash
sudo modprobe nvidia-uvm
```

To make it persist across reboots:

```bash
echo "nvidia-uvm" | sudo tee /etc/modules-load.d/nvidia-uvm.conf
```

## 3. User Permissions

The user running voicebox needs access to the GPU devices:

```bash
# Check which group owns the devices
ls -la /dev/nvidia0
# Usually: crw-rw---- 1 root video ...

# Add your user (or the voicebox service user) to that group
sudo usermod -aG video $USER
sudo usermod -aG render $USER

# Log out and back in for group changes to take effect
```

## 4a. Bare Metal Install

The setup script handles everything:

```bash
sudo ./setup-linux.sh check    # verify GPU is detected
sudo ./setup-linux.sh install  # installs with CUDA PyTorch
```

PyTorch bundles its own CUDA runtime, so you do **not** need to install the CUDA toolkit separately.

## 4b. Docker Install

### Install NVIDIA Container Toolkit

```bash
# Add the repo
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
```

### Configure Docker Runtime

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Verify GPU Access in Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Run Voicebox

```bash
cd backend
docker compose up -d
```

The `docker-compose.yml` already requests GPU access. Verify:

```bash
curl http://localhost:17493/health
# Should show: "gpu_available": true, "gpu_type": "CUDA"
```

## Troubleshooting

### `nvidia-smi` works but PyTorch can't see GPU

```bash
# Inside the container or venv:
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If `False`, the CUDA version bundled with PyTorch may not match your driver. Check compatibility:

| Driver Version | Max CUDA Version |
|---------------|-----------------|
| 525.x         | 12.0            |
| 535.x         | 12.2            |
| 545.x         | 12.3            |
| 550.x+        | 12.4            |

Reinstall PyTorch with the right CUDA version if needed:

```bash
# Example for CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Permission denied on `/dev/nvidia*`

```bash
# Quick fix (non-persistent)
sudo chmod 666 /dev/nvidia*

# Proper fix — add user to video group
sudo usermod -aG video voicebox
sudo systemctl restart voicebox
```

### Docker: `could not select device driver "nvidia"`

The NVIDIA Container Toolkit is not installed or Docker wasn't restarted:

```bash
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Out of memory (OOM)

The 1.7B model needs ~4GB VRAM. The 0.6B model needs ~2GB. Check usage:

```bash
nvidia-smi

# Use the smaller model via the API:
curl -X POST http://localhost:17493/generate \
  -H "Content-Type: application/json" \
  -d '{"profile_id": "...", "text": "hello", "model_size": "0.6B"}'
```

### Models not downloading

HuggingFace downloads go to `$HF_HOME` (default: `~/.cache/huggingface`). In Docker this is `/data/huggingface` inside the volume.

If downloads fail behind a proxy:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HTTPS_PROXY=http://your-proxy:port
```
