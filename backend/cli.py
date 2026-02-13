#!/usr/bin/env python3
"""CLI for voicebox — headless TTS generation and voice management."""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

DEFAULT_URL = "http://127.0.0.1:17493"
SERVER_BIN = "/Applications/Voicebox.app/Contents/MacOS/voicebox-server"
DEFAULT_DATA_DIR = Path.home() / "Library/Application Support/sh.voicebox.app"


# --- API helpers ---

def api(method, base_url, path, **kwargs):
    """Make an API call with consistent error handling."""
    kwargs.setdefault("timeout", 30)
    try:
        resp = getattr(requests, method)(f"{base_url}{path}", **kwargs)
    except requests.ConnectionError:
        print(f"Error: cannot connect to server at {base_url}", file=sys.stderr)
        print("Start it with: voicebox server", file=sys.stderr)
        sys.exit(1)
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text[:200])
        except Exception:
            detail = resp.text[:200]
        print(f"Error: {resp.status_code} — {detail}", file=sys.stderr)
        sys.exit(1)
    return resp


# --- Subcommands ---

PID_FILE = Path.home() / ".voicebox.pid"
LOG_FILE = Path.home() / ".voicebox.log"


def cmd_server(args):
    """Start the backend server (no frontend)."""
    data_dir = args.data_dir or str(DEFAULT_DATA_DIR)
    port = str(args.port)

    if args.stop:
        _stop_server()
        return

    # Try the installed app binary first
    if Path(SERVER_BIN).exists():
        bin_path = SERVER_BIN
    elif shutil.which("voicebox-server"):
        bin_path = "voicebox-server"
    else:
        bin_path = None

    cmd = ([bin_path, "--data-dir", data_dir, "--port", port] if bin_path
           else [sys.executable, "-m", "backend.main", "--port", port])

    if args.detach:
        # Check if already running
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text().strip())
            try:
                import os
                os.kill(pid, 0)
                print(f"Server already running (pid {pid})")
                return
            except ProcessLookupError:
                PID_FILE.unlink(missing_ok=True)

        log = open(LOG_FILE, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=log,
            start_new_session=True,
            cwd=None if bin_path else Path(__file__).resolve().parent.parent,
        )
        PID_FILE.write_text(str(proc.pid))
        print(f"Waiting for server (pid {proc.pid}, port {port})...", end="", flush=True)
        url = f"http://127.0.0.1:{port}/health"
        for _ in range(60):
            time.sleep(0.5)
            if proc.poll() is not None:
                print(" failed.")
                print(f"Server exited. Check log: {LOG_FILE}", file=sys.stderr)
                PID_FILE.unlink(missing_ok=True)
                sys.exit(1)
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    print(" ready.")
                    print(f"Stop with: voicebox server --stop")
                    return
            except requests.ConnectionError:
                print(".", end="", flush=True)
        print(" timed out.")
        print(f"Server didn't respond within 30s. Check log: {LOG_FILE}", file=sys.stderr)
    else:
        print(f"Starting voicebox server on port {port}...")
        print(f"Data dir: {data_dir}")
        print(f"Press Ctrl+C to stop.\n")
        try:
            subprocess.run(
                cmd,
                cwd=None if bin_path else Path(__file__).resolve().parent.parent,
            )
        except KeyboardInterrupt:
            print("\nServer stopped.")


def _stop_server():
    """Stop a detached server."""
    import os, signal
    if not PID_FILE.exists():
        print("No server running (no pid file).")
        return
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped server (pid {pid})")
    except ProcessLookupError:
        print(f"Server not running (stale pid {pid})")
    PID_FILE.unlink(missing_ok=True)


def cmd_voices(args):
    """List all voice profiles."""
    resp = api("get", args.url, "/profiles")
    profiles = resp.json()
    if not profiles:
        print("No voice profiles found. Import one with: voicebox import <file.zip>")
        return
    print(f"{'Name':<30} {'Language':<10} {'ID'}")
    print("-" * 75)
    for p in profiles:
        print(f"{p['name']:<30} {p['language']:<10} {p['id']}")


def cmd_import(args):
    """Import a voice profile from a ZIP file."""
    zip_path = Path(args.file)
    if not zip_path.exists():
        print(f"Error: file not found: {zip_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Importing {zip_path.name}...")
    with open(zip_path, "rb") as f:
        resp = api("post", args.url, "/profiles/import",
                    files={"file": (zip_path.name, f, "application/zip")},
                    timeout=60)
    profile = resp.json()
    print(f"Imported: {profile['name']} ({profile['id']})")


def cmd_generate(args):
    """Generate speech from text."""
    # Resolve text input
    if args.text:
        text = args.text
    elif args.file:
        text = Path(args.file).read_text().strip()
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        print("Error: provide text via --text, --file, or stdin.", file=sys.stderr)
        sys.exit(1)

    if not text:
        print("Error: text is empty.", file=sys.stderr)
        sys.exit(1)

    # Resolve voice
    profile = resolve_profile(args.url, args.voice)

    # Generate
    payload = {
        "profile_id": profile["id"],
        "text": text,
        "language": args.language or profile.get("language", "en"),
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.instruct:
        payload["instruct"] = args.instruct

    print(f"Generating with voice '{profile['name']}'...")
    start = time.time()
    resp = api("post", args.url, "/generate", json=payload, timeout=300)
    result = resp.json()
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s (audio duration: {result['duration']:.1f}s)")

    # Download wav then convert to m4a
    tag = str(int(time.time()))[-5:]
    wav_path = f"output_{tag}.wav"
    dl = api("get", args.url, f"/audio/{result['id']}", timeout=60)
    Path(wav_path).write_bytes(dl.content)

    if not shutil.which("ffmpeg"):
        print("Warning: ffmpeg not found, keeping .wav", file=sys.stderr)
        output = wav_path
    else:
        output = args.output or f"output_{tag}.m4a"
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-c:a", "aac", "-b:a", "128k", output],
            capture_output=True,
        )
        if r.returncode != 0:
            print(f"Warning: ffmpeg failed, keeping .wav", file=sys.stderr)
            output = wav_path
        else:
            Path(wav_path).unlink()
            print(f"Saved: {output}")

    if not args.no_open:
        subprocess.run(["open", output])


def cmd_health(args):
    """Check server health."""
    resp = api("get", args.url, "/health")
    h = resp.json()
    print(f"Status:       {h['status']}")
    print(f"Model loaded: {h['model_loaded']}")
    print(f"Backend:      {h.get('backend_type', '?')}")
    print(f"GPU:          {h.get('gpu_type', 'none')}")
    if h.get('vram_used_mb'):
        print(f"VRAM used:    {h['vram_used_mb']:.0f} MB")


# --- Profile resolution (shared) ---

def resolve_profile(base_url, voice_name):
    resp = api("get", base_url, "/profiles")
    profiles = resp.json()
    if not profiles:
        print("Error: no voice profiles found.", file=sys.stderr)
        print("Import one with: voicebox import <file.zip>", file=sys.stderr)
        sys.exit(1)

    if voice_name:
        match = [p for p in profiles if p["name"].lower() == voice_name.lower()]
        if not match:
            match = [p for p in profiles if voice_name.lower() in p["name"].lower()]
        if not match:
            print(f"Error: no voice matching '{voice_name}'. Available:", file=sys.stderr)
            for p in profiles:
                print(f"  - {p['name']}", file=sys.stderr)
            sys.exit(1)
        if len(match) > 1:
            print(f"Multiple voices match '{voice_name}':", file=sys.stderr)
            for p in match:
                print(f"  - {p['name']}", file=sys.stderr)
            sys.exit(1)
        return match[0]
    else:
        # Interactive picker
        print("Available voices:")
        for i, p in enumerate(profiles, 1):
            print(f"  {i}. {p['name']} ({p['language']})")
        print()
        while True:
            choice = input(f"Choose a voice [1-{len(profiles)}]: ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(profiles):
                    return profiles[idx]
            except ValueError:
                pass
            print("Invalid choice, try again.")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(prog="voicebox", description="Voicebox CLI — headless TTS")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Server URL (default: {DEFAULT_URL})")
    sub = parser.add_subparsers(dest="command")

    # server
    p_server = sub.add_parser("server", help="Start the backend server")
    p_server.add_argument("--port", type=int, default=17493, help="Port (default: 17493)")
    p_server.add_argument("--data-dir", help="Data directory")
    p_server.add_argument("-d", "--detach", action="store_true", help="Run in background (daemon)")
    p_server.add_argument("--stop", action="store_true", help="Stop a detached server")

    # voices
    sub.add_parser("voices", help="List voice profiles")

    # import
    p_import = sub.add_parser("import", help="Import a voice profile from ZIP")
    p_import.add_argument("file", help="Path to .zip file")

    # generate
    p_gen = sub.add_parser("generate", aliases=["gen", "say"], help="Generate speech")
    p_gen.add_argument("--voice", "-v", help="Voice name (interactive picker if omitted)")
    p_gen.add_argument("--text", "-t", help="Text to speak")
    p_gen.add_argument("--file", "-f", help="Read text from a file")
    p_gen.add_argument("--output", "-o", help="Output path (default: output_<epoch>.m4a)")
    p_gen.add_argument("--language", "-l", help="Language code")
    p_gen.add_argument("--seed", "-s", type=int, help="Random seed")
    p_gen.add_argument("--instruct", help="Style instruction (e.g. 'speak slowly')")
    p_gen.add_argument("--no-open", action="store_true", help="Don't open file after generating")

    # health
    sub.add_parser("health", help="Check server status")

    args = parser.parse_args()

    if args.command == "server":
        cmd_server(args)
    elif args.command == "voices":
        cmd_voices(args)
    elif args.command == "import":
        cmd_import(args)
    elif args.command in ("generate", "gen", "say"):
        cmd_generate(args)
    elif args.command == "health":
        cmd_health(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
