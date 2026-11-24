"""
Torch floor for CVE-2026-24747: torch < 2.10 lets a crafted checkpoint run
code during torch.load even with weights_only=True.

Every platform with 2.10+ wheels must require them. Intel macOS stays on
2.2.x (no newer x86_64 macOS wheels exist) and Windows ROCm stays on AMD's
2.9.1 wheel until a stable 2.10 build ships.
"""

from pathlib import Path

import pytest
from packaging.requirements import Requirement

BACKEND_DIR = Path(__file__).parent.parent
REPO_DIR = BACKEND_DIR.parent

PLATFORMS = {
    "linux-x86_64": {"sys_platform": "linux", "platform_machine": "x86_64"},
    "linux-aarch64": {"sys_platform": "linux", "platform_machine": "aarch64"},
    "windows-x86_64": {"sys_platform": "win32", "platform_machine": "AMD64"},
    "macos-arm64": {"sys_platform": "darwin", "platform_machine": "arm64"},
    "macos-x86_64": {"sys_platform": "darwin", "platform_machine": "x86_64"},
}

# (first patched version, newest vulnerable version)
FLOORS = {
    "torch": ("2.10.0", "2.9.1"),
    "torchaudio": ("2.10.0", "2.9.1"),
    "torchvision": ("0.25.0", "0.24.1"),
}


def _requirements(path: Path) -> list[Requirement]:
    reqs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")) or " @ " in line:
            continue
        reqs.append(Requirement(line))
    return reqs


def _active(path: Path, name: str, platform: str) -> list[Requirement]:
    env = PLATFORMS[platform]
    return [r for r in _requirements(path) if r.name == name and (r.marker is None or r.marker.evaluate(env))]


@pytest.mark.parametrize(
    "path,name",
    [
        (BACKEND_DIR / "requirements.txt", "torch"),
        (BACKEND_DIR / "requirements.txt", "torchaudio"),
        (REPO_DIR / "requirements.txt", "torch"),
        (REPO_DIR / "requirements.txt", "torchvision"),
    ],
)
@pytest.mark.parametrize("platform", [p for p in PLATFORMS if p != "macos-x86_64"])
def test_patched_torch_required(path, name, platform):
    active = _active(path, name, platform)
    assert len(active) == 1, f"{path}: expected one {name} requirement on {platform}, got {active}"
    patched, vulnerable = FLOORS[name]
    assert active[0].specifier.contains(patched)
    assert not active[0].specifier.contains(vulnerable), f"{active[0]} still allows {name} {vulnerable}"


@pytest.mark.parametrize(
    "path,name,version",
    [
        (BACKEND_DIR / "requirements.txt", "torch", "2.2.2"),
        (BACKEND_DIR / "requirements.txt", "torchaudio", "2.2.2"),
        (REPO_DIR / "requirements.txt", "torch", "2.2.2"),
        (REPO_DIR / "requirements.txt", "torchvision", "0.17.2"),
    ],
)
def test_intel_macos_keeps_last_available_torch(path, name, version):
    active = _active(path, name, "macos-x86_64")
    assert len(active) == 1
    assert active[0].specifier.contains(version)


def test_windows_rocm_pin_is_documented_exception():
    content = (BACKEND_DIR / "requirements-rocm.txt").read_text()
    assert "torch==2.9.1+rocm7.2.1" in content
    assert "CVE-2026-24747" in content
