"""
End-to-end model generation test.

Exercises every TTS model against the frozen PyInstaller binary, captures
per-model pass/fail, and writes a JSON + Markdown report.

Usage:
    python backend/tests/test_all_models_e2e.py [flags]

See E2E_MODEL_TEST_DESIGN.md for the full design.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
DIST_DIR = BACKEND_DIR / "dist"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── Test matrix ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class MatrixRow:
    label: str            # human-readable (appears in report)
    engine: str           # /generate engine
    model_size: Optional[str]  # /generate model_size (None = omit)
    profile_kind: str     # "cloned" | "preset_kokoro" | "preset_qwen_cv"
    model_name: str       # /models/status key for cache lookup


MATRIX: list[MatrixRow] = [
    MatrixRow("qwen 1.7B",              "qwen",              "1.7B", "cloned",          "qwen-tts-1.7B"),
    MatrixRow("qwen 0.6B",              "qwen",              "0.6B", "cloned",          "qwen-tts-0.6B"),
    MatrixRow("qwen_custom_voice 1.7B", "qwen_custom_voice", "1.7B", "preset_qwen_cv",  "qwen-custom-voice-1.7B"),
    MatrixRow("qwen_custom_voice 0.6B", "qwen_custom_voice", "0.6B", "preset_qwen_cv",  "qwen-custom-voice-0.6B"),
    MatrixRow("luxtts",                 "luxtts",            None,   "cloned",          "luxtts"),
    MatrixRow("chatterbox",             "chatterbox",        None,   "cloned",          "chatterbox-tts"),
    MatrixRow("chatterbox_turbo",       "chatterbox_turbo",  None,   "cloned",          "chatterbox-turbo"),
    MatrixRow("tada 1B",                "tada",              "1B",   "cloned",          "tada-1b"),
    MatrixRow("tada 3B",                "tada",              "3B",   "cloned",          "tada-3b-ml"),
    MatrixRow("kokoro",                 "kokoro",            None,   "preset_kokoro",   "kokoro"),
]

TEXT = "The quick brown fox jumps over the lazy dog."
DEFAULT_TIMEOUT_CACHED = 180
DEFAULT_TIMEOUT_DOWNLOAD = 1200
HEALTH_TIMEOUT = 120


# ── Result record ────────────────────────────────────────────────────

@dataclass
class ModelResult:
    label: str
    engine: str
    model_size: Optional[str]
    status: str                      # "passed" | "failed" | "timeout"
    was_cached: Optional[bool] = None
    generation_id: Optional[str] = None
    elapsed_seconds: float = 0.0
    audio_duration: Optional[float] = None
    audio_path: Optional[str] = None
    audio_bytes: Optional[int] = None
    error: Optional[str] = None
    http_status: Optional[int] = None
    server_log_tail: Optional[list[str]] = None


# ── Binary resolution ────────────────────────────────────────────────

def find_binary() -> Optional[Path]:
    """Return the first existing binary in priority order, or None."""
    is_win = platform.system() == "Windows"
    exe = ".exe" if is_win else ""
    candidates = [
        DIST_DIR / "voicebox-server-cuda" / f"voicebox-server-cuda{exe}",
        DIST_DIR / f"voicebox-server{exe}",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def build_binary() -> Path:
    """Invoke build_binary.py and return the resulting binary path."""
    print("[build] No frozen binary found — invoking build_binary.py (this may take 5-20 minutes)...", flush=True)
    script = BACKEND_DIR / "build_binary.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(BACKEND_DIR),
    )
    if result.returncode != 0:
        raise RuntimeError(f"build_binary.py exited with code {result.returncode}")
    found = find_binary()
    if found is None:
        raise RuntimeError("build_binary.py finished but no binary was found in backend/dist/")
    return found


# ── Server spawn + log capture ───────────────────────────────────────

class ServerProcess:
    def __init__(self, binary: Path, port: int, data_dir: Path, log_path: Path):
        self.binary = binary
        self.port = port
        self.data_dir = data_dir
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None
        self._log_buffer: deque[str] = deque(maxlen=500)
        self._reader_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        args = [
            str(self.binary),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--data-dir", str(self.data_dir),
            "--parent-pid", str(os.getpid()),
        ]
        print(f"[spawn] {' '.join(args)}", flush=True)
        self._log_fh = open(self.log_path, "w", encoding="utf-8", errors="replace")
        # Combine stderr into stdout so we get a single ordered stream.
        self.proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            errors="replace",
        )
        self._reader_thread = threading.Thread(target=self._pump_logs, daemon=True)
        self._reader_thread.start()

    def _pump_logs(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            self._log_buffer.append(line.rstrip("\n"))
            self._log_fh.write(line)
            self._log_fh.flush()

    def log_tail(self, n: int = 100) -> list[str]:
        tail = list(self._log_buffer)[-n:]
        return tail

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is not None:
            return
        try:
            if platform.system() == "Windows":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                    capture_output=True,
                )
            else:
                self.proc.send_signal(signal.SIGTERM)
        except Exception as e:
            print(f"[shutdown] signal failed: {e}", flush=True)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("[shutdown] server didn't exit cleanly, killing", flush=True)
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)
        try:
            self._log_fh.close()
        except Exception:
            pass


def pick_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ── HTTP helpers ─────────────────────────────────────────────────────

def wait_for_health(base_url: str, server: ServerProcess, timeout: int) -> None:
    deadline = time.time() + timeout
    with httpx.Client(timeout=5.0) as client:
        while time.time() < deadline:
            if not server.is_alive():
                raise RuntimeError("Server process exited before becoming healthy")
            try:
                r = client.get(f"{base_url}/health")
                if r.status_code == 200 and r.json().get("status") == "healthy":
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1.0)
    raise TimeoutError(f"Server did not become healthy within {timeout}s")


def get_model_cached(client: httpx.Client, base_url: str, model_name: str) -> Optional[bool]:
    try:
        r = client.get(f"{base_url}/models/status", timeout=30.0)
        r.raise_for_status()
        for m in r.json().get("models", []):
            if m.get("model_name") == model_name:
                return bool(m.get("downloaded"))
    except httpx.HTTPError:
        return None
    return None


def create_cloned_profile(client: httpx.Client, base_url: str, wav_path: Path, reference_text: str) -> str:
    r = client.post(f"{base_url}/profiles", json={
        "name": "e2e-cloned",
        "voice_type": "cloned",
        "language": "en",
    })
    r.raise_for_status()
    profile_id = r.json()["id"]

    with open(wav_path, "rb") as f:
        r = client.post(
            f"{base_url}/profiles/{profile_id}/samples",
            files={"file": (wav_path.name, f, "audio/wav")},
            data={"reference_text": reference_text},
            timeout=120.0,
        )
    r.raise_for_status()
    return profile_id


def create_preset_profile(client: httpx.Client, base_url: str, name: str, engine: str, voice_id: str) -> str:
    r = client.post(f"{base_url}/profiles", json={
        "name": name,
        "voice_type": "preset",
        "language": "en",
        "preset_engine": engine,
        "preset_voice_id": voice_id,
    })
    r.raise_for_status()
    return r.json()["id"]


def run_one_generation(
    client: httpx.Client,
    base_url: str,
    row: MatrixRow,
    profile_id: str,
    timeout_s: int,
) -> tuple[str, dict]:
    """Start a generation and stream its status until done/failed/timeout.

    Returns (status, payload) where status is "completed" | "failed" | "timeout".
    """
    body = {
        "profile_id": profile_id,
        "text": TEXT,
        "language": "en",
        "engine": row.engine,
        "seed": 42,
        "normalize": True,
    }
    if row.model_size is not None:
        body["model_size"] = row.model_size

    r = client.post(f"{base_url}/generate", json=body, timeout=30.0)
    r.raise_for_status()
    gen = r.json()
    gen_id = gen["id"]

    deadline = time.time() + timeout_s
    last_payload: dict = gen
    status_url = f"{base_url}/generate/{gen_id}/status"

    while time.time() < deadline:
        remaining = max(1.0, deadline - time.time())
        try:
            with client.stream("GET", status_url, timeout=httpx.Timeout(remaining + 5, read=remaining + 5)) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        payload = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    last_payload = payload
                    status = payload.get("status")
                    if status == "not_found":
                        return "failed", {"error": "generation not found", **payload}
                    if status in ("completed", "failed"):
                        return status, payload
                    if time.time() >= deadline:
                        break
        except httpx.HTTPError:
            time.sleep(1.0)
            continue

    return "timeout", last_payload


def fetch_audio_info(
    client: httpx.Client, base_url: str, generation_id: str, data_dir: Path
) -> tuple[Optional[str], Optional[int]]:
    """Return (audio_path, audio_bytes) for a completed generation.

    Server stores audio_path relative to data_dir; resolve it to get a size.
    """
    try:
        r = client.get(f"{base_url}/history/{generation_id}", timeout=10.0)
        if r.status_code != 200:
            return None, None
        data = r.json()
        audio_path = data.get("audio_path")
        if not audio_path:
            return None, None
        p = Path(audio_path)
        if not p.is_absolute():
            p = data_dir / p
        if p.exists():
            return str(p), p.stat().st_size
        return audio_path, None
    except httpx.HTTPError:
        return None, None


# ── Report writers ───────────────────────────────────────────────────

def write_reports(
    output_dir: Path,
    binary: Path,
    started_at: datetime,
    finished_at: datetime,
    results: list[ModelResult],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plat = f"{platform.system().lower()}-{platform.machine().lower()}"
    ts = started_at.strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"e2e-{plat}-{ts}.json"
    md_path = output_dir / f"e2e-{plat}-{ts}.md"

    doc = {
        "platform": plat,
        "binary": str(binary),
        "binary_size_mb": round(binary.stat().st_size / (1024 * 1024), 1) if binary.exists() else None,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": (finished_at - started_at).total_seconds(),
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(doc, indent=2))

    lines = [
        f"# Voicebox E2E — {plat} — {started_at.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"Binary: `{binary}`  ",
        f"Elapsed: {doc['elapsed_seconds']:.1f}s",
        "",
        "| Model | Status | Cached | Elapsed | Audio | Error |",
        "|-------|--------|--------|---------|-------|-------|",
    ]
    for r in results:
        status_icon = {"passed": "PASS", "failed": "FAIL", "timeout": "TIMEOUT"}.get(r.status, r.status.upper())
        cached = "yes" if r.was_cached else ("no" if r.was_cached is False else "?")
        audio_col = f"{r.audio_duration:.2f}s" if r.audio_duration else ("—" if r.status != "passed" else "?")
        error_col = (r.error or "").replace("\n", " ")[:120]
        lines.append(f"| {r.label} | {status_icon} | {cached} | {r.elapsed_seconds:.1f}s | {audio_col} | {error_col} |")

    failed_rows = [r for r in results if r.status != "passed"]
    if failed_rows:
        lines.append("")
        lines.append("## Failures")
        for r in failed_rows:
            lines.append("")
            lines.append(f"### {r.label} — {r.status}")
            if r.error:
                lines.append("")
                lines.append("```")
                lines.append(r.error)
                lines.append("```")
            if r.server_log_tail:
                lines.append("")
                lines.append("<details><summary>server log (last lines)</summary>")
                lines.append("")
                lines.append("```")
                lines.extend(r.server_log_tail)
                lines.append("```")
                lines.append("</details>")

    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


# ── Main ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Voicebox E2E model generation test")
    p.add_argument("--binary", type=Path, help="Path to voicebox-server binary (overrides auto-detect)")
    p.add_argument("--skip-build", action="store_true", help="Error if binary missing instead of building")
    p.add_argument(
        "--reference-wav",
        type=Path,
        default=FIXTURES_DIR / "reference_voice.wav",
        help="Reference audio for cloning engines",
    )
    p.add_argument(
        "--reference-text",
        help="Transcription of reference-wav (default: read from fixtures/reference_voice.txt)",
    )
    p.add_argument("--only", help="Comma-separated engines to run (e.g. kokoro,qwen)")
    p.add_argument("--skip", help="Comma-separated engines to skip")
    p.add_argument("--keep-data-dir", action="store_true", help="Don't delete tempdir after run")
    p.add_argument("--timeout-cached", type=int, default=DEFAULT_TIMEOUT_CACHED)
    p.add_argument("--timeout-download", type=int, default=DEFAULT_TIMEOUT_DOWNLOAD)
    p.add_argument("--port", type=int, help="Override auto-picked port")
    p.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return p.parse_args()


def filter_matrix(args: argparse.Namespace) -> list[MatrixRow]:
    only = set(x.strip() for x in args.only.split(",")) if args.only else None
    skip = set(x.strip() for x in args.skip.split(",")) if args.skip else set()
    rows = []
    for r in MATRIX:
        if only is not None and r.engine not in only:
            continue
        if r.engine in skip:
            continue
        rows.append(r)
    return rows


def resolve_reference(args: argparse.Namespace) -> tuple[Path, str]:
    wav = args.reference_wav
    if not wav.exists():
        raise FileNotFoundError(
            f"Reference WAV not found: {wav}\n"
            f"Place a sample at {FIXTURES_DIR / 'reference_voice.wav'} or pass --reference-wav.\n"
            f"See backend/tests/fixtures/README.md."
        )
    if args.reference_text:
        text = args.reference_text
    else:
        txt_path = wav.with_suffix(".txt")
        if not txt_path.exists():
            raise FileNotFoundError(
                f"Reference transcription not found: {txt_path}\n"
                f"Create it next to the WAV, or pass --reference-text."
            )
        text = txt_path.read_text().strip()
    if not text:
        raise ValueError("Reference transcription is empty")
    return wav, text


def main() -> int:
    args = parse_args()
    rows = filter_matrix(args)
    if not rows:
        print("No rows selected after --only/--skip filtering", file=sys.stderr)
        return 2

    # Binary
    binary = args.binary or find_binary()
    if binary is None:
        if args.skip_build:
            print("No frozen binary found and --skip-build set. Run: python backend/build_binary.py", file=sys.stderr)
            return 2
        binary = build_binary()
    if not binary.exists():
        print(f"Binary path does not exist: {binary}", file=sys.stderr)
        return 2
    print(f"[binary] {binary}", flush=True)

    # Reference audio (only required if any cloning row is in the matrix)
    needs_reference = any(r.profile_kind == "cloned" for r in rows)
    ref_wav: Optional[Path] = None
    ref_text: Optional[str] = None
    if needs_reference:
        try:
            ref_wav, ref_text = resolve_reference(args)
        except (FileNotFoundError, ValueError) as e:
            print(f"[fixture] {e}", file=sys.stderr)
            return 2
        print(f"[fixture] reference WAV:  {ref_wav}", flush=True)
        print(f"[fixture] reference text: {ref_text!r}", flush=True)

    # Tempdir + log path
    data_dir = Path(tempfile.mkdtemp(prefix="voicebox-e2e-"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_path = args.output_dir / f"server-{ts}.log"

    port = args.port or pick_free_port()
    base_url = f"http://127.0.0.1:{port}"

    server = ServerProcess(binary=binary, port=port, data_dir=data_dir, log_path=log_path)
    started_at = datetime.now(timezone.utc)
    results: list[ModelResult] = []

    try:
        server.start()
        print(f"[health] waiting for {base_url}/health ...", flush=True)
        wait_for_health(base_url, server, HEALTH_TIMEOUT)
        print("[health] ready", flush=True)

        with httpx.Client(timeout=30.0) as client:
            # Profile setup (only create what's needed)
            cloned_profile_id: Optional[str] = None
            kokoro_profile_id: Optional[str] = None
            qwen_cv_profile_id: Optional[str] = None
            needed_kinds = {r.profile_kind for r in rows}
            if "cloned" in needed_kinds:
                assert ref_wav is not None and ref_text is not None
                print("[profile] creating cloned profile...", flush=True)
                cloned_profile_id = create_cloned_profile(client, base_url, ref_wav, ref_text)
            if "preset_kokoro" in needed_kinds:
                print("[profile] creating kokoro preset...", flush=True)
                kokoro_profile_id = create_preset_profile(client, base_url, "e2e-kokoro", "kokoro", "af_heart")
            if "preset_qwen_cv" in needed_kinds:
                print("[profile] creating qwen_custom_voice preset...", flush=True)
                qwen_cv_profile_id = create_preset_profile(client, base_url, "e2e-qwen-cv", "qwen_custom_voice", "Ryan")

            profile_lookup = {
                "cloned": cloned_profile_id,
                "preset_kokoro": kokoro_profile_id,
                "preset_qwen_cv": qwen_cv_profile_id,
            }

            # Matrix loop
            for row in rows:
                print(f"\n[run] {row.label} (engine={row.engine}, size={row.model_size})", flush=True)
                profile_id = profile_lookup[row.profile_kind]
                assert profile_id is not None
                was_cached = get_model_cached(client, base_url, row.model_name)
                timeout_s = args.timeout_cached if was_cached else args.timeout_download
                print(f"[run] cached={was_cached} timeout={timeout_s}s", flush=True)

                t0 = time.time()
                result = ModelResult(
                    label=row.label,
                    engine=row.engine,
                    model_size=row.model_size,
                    status="failed",
                    was_cached=was_cached,
                )
                try:
                    status, payload = run_one_generation(client, base_url, row, profile_id, timeout_s)
                    result.status = "passed" if status == "completed" else status
                    result.generation_id = payload.get("id")
                    result.audio_duration = payload.get("duration")
                    result.error = payload.get("error")
                    if status == "completed" and result.generation_id:
                        audio_path, audio_bytes = fetch_audio_info(
                            client, base_url, result.generation_id, data_dir
                        )
                        result.audio_path = audio_path
                        result.audio_bytes = audio_bytes
                        if audio_bytes is not None and audio_bytes == 0:
                            result.status = "failed"
                            result.error = (result.error or "") + " (audio file is empty)"
                except httpx.HTTPStatusError as e:
                    result.status = "failed"
                    result.http_status = e.response.status_code
                    try:
                        detail = e.response.json().get("detail")
                    except Exception:
                        detail = e.response.text
                    result.error = f"HTTP {e.response.status_code}: {detail}"
                except Exception as e:
                    result.status = "failed"
                    result.error = f"{type(e).__name__}: {e}"

                result.elapsed_seconds = round(time.time() - t0, 2)
                if result.status != "passed":
                    result.server_log_tail = server.log_tail(100)
                print(f"[run] {row.label} → {result.status} in {result.elapsed_seconds}s"
                      + (f" ({result.error})" if result.error else ""), flush=True)
                results.append(result)
    finally:
        finished_at = datetime.now(timezone.utc)
        server.stop()
        if not args.keep_data_dir:
            shutil.rmtree(data_dir, ignore_errors=True)
        else:
            print(f"[cleanup] keeping data dir: {data_dir}", flush=True)

    json_path, md_path = write_reports(args.output_dir, binary, started_at, finished_at, results)
    print(f"\n[report] {json_path}")
    print(f"[report] {md_path}")
    print(f"[report] server log: {log_path}")

    passed = sum(1 for r in results if r.status == "passed")
    failed = len(results) - passed
    print(f"\n== {passed} passed, {failed} failed ==")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
