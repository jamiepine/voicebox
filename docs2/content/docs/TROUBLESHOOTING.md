---
title: "Troubleshooting Guide"
description: "Common issues and solutions for Voicebox"
---

Common issues and solutions for Voicebox.

## Installation Issues

### macOS: "Voicebox cannot be opened because it is from an unidentified developer"

**Solution:**

1. Right-click the `.dmg` file
2. Select "Open"
3. Click "Open" in the security dialog
4. Alternatively, go to System Settings → Privacy & Security → Allow Voicebox

### Windows: "Windows protected your PC"

**Solution:**

1. Click "More info"
2. Click "Run anyway"
3. Windows Defender may flag new software; this is normal for unsigned apps

### Linux: AppImage won't run

**Solution:**

```bash
chmod +x voicebox-*.AppImage
./voicebox-*.AppImage
```

## Runtime Issues

### Server won't start

**Symptoms:** App opens but shows "Server not connected"

**Solutions:**

1. **Check Python installation**

   ```bash
   python --version  # Should be 3.11+
   ```

2. **Check server binary exists**

   - Look in `tauri/src-tauri/binaries/` for your platform
   - Binary should match your system architecture

3. **Check permissions**

   ```bash
   # macOS/Linux
   chmod +x tauri/src-tauri/binaries/voicebox-server-*
   ```

4. **Check logs**
   - macOS: Open Console.app and search for "voicebox"
   - Linux: Check `~/.local/share/voicebox/` for logs
   - Windows: Check Event Viewer

### "Model download failed"

**Symptoms:** First generation fails with download error

**Solutions:**

1. **Check internet connection**

   - Models download from HuggingFace Hub (~2-4GB)
   - First download may take several minutes

2. **Check disk space**

   - Models are cached in `~/.cache/huggingface/`
   - Ensure at least 5GB free space

3. **Manual download** (if automatic fails)
   ```bash
   pip install huggingface_hub
   huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base
   ```

### "Out of memory" errors

**Symptoms:** Generation fails with CUDA/VRAM errors

**Solutions:**

1. **Use smaller model**

   - Switch to 0.6B model instead of 1.7B
   - Settings → Model Management → Load 0.6B

2. **Close other applications**

   - Free up GPU memory
   - Close browser tabs, other ML apps

3. **Use CPU mode**
   - Slower but works without GPU
   - Backend automatically falls back to CPU

### MLX "Failed to load the default metallib" error (Apple Silicon)

**Symptoms:** Generation fails with "library not found" or "metallib" errors

**Solutions:**

1. **Rebuild server binary**

   ```bash
   bun run build:server
   ```

   The build script should automatically include MLX Metal shader libraries.

2. **Check MLX installation**

   ```bash
   pip install -r backend/requirements-mlx.txt
   ```

3. **Verify backend detection**
   - Check server logs for "Backend: MLX"
   - If showing "Backend: PYTORCH", MLX may not be installed correctly

### Audio playback issues

**Symptoms:** Generated audio won't play

**Solutions:**

1. **Check audio format**

   - Audio is saved as WAV files
   - Ensure your system supports WAV playback

2. **Try downloading audio**

   - Right-click → Download
   - Play in external player

3. **Check browser permissions** (web version)
   - Allow audio autoplay in browser settings

### Slow generation

**Symptoms:** Generation takes >30 seconds

**Solutions:**

1. **Check backend type** (Apple Silicon)

   - Check Settings → Server Status
   - Should show "Backend: MLX" on Apple Silicon
   - If showing "Backend: PYTORCH", install MLX: `pip install -r backend/requirements-mlx.txt`
   - MLX provides 4-5x faster inference on Apple Silicon

2. **Use GPU** (if available)

   - Check Settings → Server Status
   - Should show "GPU available: true"
   - Apple Silicon: Should show "Metal (Apple Silicon via MLX)"
   - Windows/Linux: Should show "CUDA" if GPU available

3. **Enable caching**

   - Voice prompts are cached automatically
   - Second generation with same voice should be faster

4. **Use smaller model**

   - 0.6B model is faster than 1.7B
   - Quality difference is minimal for most voices

5. **Check system resources**
   - Close other CPU/GPU intensive apps
   - Ensure adequate RAM (8GB+ recommended)

## API Issues

### "Connection refused" when using API

**Solutions:**

1. **Check server is running**

   ```bash
   curl http://localhost:8000/health
   ```

2. **Check remote mode**

   - If connecting remotely, ensure server is started with `--host 0.0.0.0`
   - Check firewall settings

3. **Check port availability**
   - Default port is 8000
   - Ensure no other service is using it

### CORS errors in browser

**Solutions:**

1. **Use desktop app** (recommended)

   - Desktop app doesn't have CORS restrictions

2. **Configure CORS** (for web deployment)
   - Update `backend/main.py` CORS settings
   - Add your domain to allowed origins

## Update Issues

### "Update check failed"

**Solutions:**

1. **Check internet connection**

   - Updates are fetched from GitHub releases

2. **Check GitHub access**

   - Ensure `github.com` is accessible
   - Check firewall/proxy settings

3. **Manual update**
   - Download latest release from GitHub
   - Install manually

### "Invalid signature" error

**Solutions:**

1. **Re-download installer**

   - Signature may be corrupted
   - Download fresh copy from GitHub

2. **Check release integrity**
   - Verify `.sig` file matches installer
   - Report issue if signature is invalid

## Data Issues

### Profiles disappeared

**Solutions:**

1. **Check data directory**

   - macOS: `~/Library/Application Support/voicebox/`
   - Windows: `%APPDATA%/voicebox/`
   - Linux: `~/.local/share/voicebox/`

2. **Check database**

   - Database: `data/voicebox.db`
   - Ensure file exists and is readable

3. **Restore from backup**
   - Profiles can be exported/imported
   - Check for backup files

### "Database locked" error

**Solutions:**

1. **Close other instances**

   - Ensure only one Voicebox instance is running

2. **Restart app**

   - Close and reopen Voicebox

3. **Check file permissions**
   - Ensure database file is writable
   - Check directory permissions

## Development Issues

### Build fails

**Solutions:**

1. **Check Rust installation**

   ```bash
   rustc --version
   rustup update
   ```

2. **Check Tauri dependencies**

   ```bash
   cd tauri
   bun install
   ```

3. **Clean build**
   ```bash
   cd tauri/src-tauri
   cargo clean
   cd ../..
   bun run build
   ```

### API client generation fails

**Solutions:**

1. **Start backend server**

   ```bash
   bun run dev:server
   ```

2. **Check OpenAPI endpoint**

   ```bash
   curl http://localhost:8000/openapi.json
   ```

3. **Regenerate client**
   ```bash
   bun run generate:api
   ```

## Still Having Issues?

1. **Check existing issues**

   - Search GitHub issues for similar problems
   - Check closed issues for solutions

2. **Create new issue**

   - Include:
     - OS and version
     - Voicebox version
     - Steps to reproduce
     - Error messages/logs
     - Screenshots (if applicable)

3. **Get help**
   - Check documentation in `docs/`
   - Review `backend/README.md` for API details
   - See `CONTRIBUTING.md` for development help

---

For more help, open an issue on [GitHub](https://github.com/jamiepine/voicebox/issues).
