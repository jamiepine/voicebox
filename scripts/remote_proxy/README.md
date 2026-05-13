# Voicebox Remote Proxy Helper
Optional developer helper. Not required for normal Voicebox desktop use.
It provides a lightweight local proxy that sits between the Voicebox desktop application and an external remote backend (such as a Cloudflare Tunnel or Google Colab instance).

### Why use this?
When using Remote Server mode, the Voicebox desktop application polls the `/health` endpoint during startup to ensure the backend is ready before letting you into the app.

If your remote backend sits behind a rate-limiter (like a free Cloudflare Tunnel), the desktop app's polling loop might trigger an HTTP `429 Too Many Requests` response, effectively blocking the app from starting up. 

This proxy solves that by:
1. Intercepting the `/health` checks and returning the required JSON payload locally.
2. Transparently forwarding all other actual API requests (`/generate`, `/profiles`, etc.) to your upstream remote URL.
3. Allowing you to dynamically update the upstream Cloudflare URL via a local API call without needing to restart the app or the proxy.

### Setup

1. Create a virtual environment and install dependencies:
```powershell
py -3.12 -m venv .venv-remote-proxy
.\.venv-remote-proxy\Scripts\Activate.ps1
python -m pip install fastapi uvicorn[standard] httpx
```

2. Start the proxy with your upstream URL:
```powershell
$env:VOICEBOX_UPSTREAM_URL = "https://your-custom-url.trycloudflare.com"
python voicebox_remote_proxy.py
```

3. In the Voicebox Desktop App, configure your settings:
- **Server mode:** Remote / proxy server
- **Server URL:** `http://127.0.0.1:17493`

### Dynamically updating the Upstream URL
When your temporary tunnel URL changes, you don't need to restart anything. Update the proxy dynamically using PowerShell:
```powershell
$body = @{ upstream_url = "https://NEW-URL.trycloudflare.com" } | ConvertTo-Json

Invoke-RestMethod -Method Put -Uri "http://127.0.0.1:17493/_proxy/upstream" -ContentType "application/json" -Body $body
```
