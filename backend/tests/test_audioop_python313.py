"""
Regression tests for issue #852: audioop removed from Python 3.13 stdlib.

Voice sample validation imports audioop transitively (librosa → audioread).
The audioop-lts backport must be declared in requirements and bundled in
PyInstaller builds on 3.13+.
"""

import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from build_binary import build_server


@pytest.fixture
def backend_dir():
    return Path(__file__).parent.parent


class TestAudioopRequirements:
    def test_requirements_declare_audioop_lts_for_python_313(self, backend_dir):
        content = (backend_dir / "requirements.txt").read_text()
        assert re.search(
            r"^audioop-lts.*python_version\s*>=\s*['\"]3\.13['\"]",
            content,
            re.MULTILINE,
        ), "requirements.txt must pin audioop-lts for Python 3.13+"


@pytest.mark.skipif(sys.version_info < (3, 13), reason="Python 3.13+ only")
class TestAudioopRuntime:
    def test_audioop_importable(self):
        import audioop  # noqa: F401

    def test_validate_reference_wav_does_not_fail_on_missing_audioop(self, tmp_path):
        import numpy as np
        import soundfile as sf
        from utils.audio import validate_and_load_reference_audio

        sr = 24000
        t = np.arange(int(sr * 3), dtype=np.float32) / sr
        audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        path = tmp_path / "reference.wav"
        sf.write(str(path), audio, sr)

        ok, err, out_audio, out_sr = validate_and_load_reference_audio(str(path))

        assert ok, err
        assert out_audio is not None
        assert out_sr == sr
        assert "audioop" not in (err or "").lower()


class TestAudioopBuildArgs:
    @staticmethod
    def _hidden_imports(args):
        imports = []
        for i, arg in enumerate(args):
            if arg == "--hidden-import" and i + 1 < len(args):
                imports.append(args[i + 1])
        return imports

    def test_pyinstaller_includes_audioop_on_python_313(self):
        class FakeVersionInfo(tuple):
            @property
            def major(self):
                return self[0]

            @property
            def minor(self):
                return self[1]

            @property
            def micro(self):
                return self[2]

        fake_313 = FakeVersionInfo((3, 13, 0, "final", 0))

        with (
            patch("build_binary.PyInstaller.__main__.run") as mock_run,
            patch("build_binary.platform.system", return_value="Linux"),
            patch("build_binary.is_apple_silicon", return_value=False),
            patch("build_binary.os.chdir"),
            patch("build_binary.sys.version_info", fake_313),
        ):
            build_server()
            args = mock_run.call_args[0][0]

        assert "audioop" in self._hidden_imports(args)

    def test_pyinstaller_omits_audioop_on_python_312(self):
        class FakeVersionInfo(tuple):
            @property
            def major(self):
                return self[0]

            @property
            def minor(self):
                return self[1]

            @property
            def micro(self):
                return self[2]

        fake_312 = FakeVersionInfo((3, 12, 0, "final", 0))

        with (
            patch("build_binary.PyInstaller.__main__.run") as mock_run,
            patch("build_binary.platform.system", return_value="Linux"),
            patch("build_binary.is_apple_silicon", return_value=False),
            patch("build_binary.os.chdir"),
            patch("build_binary.sys.version_info", fake_312),
        ):
            build_server()
            args = mock_run.call_args[0][0]

        assert "audioop" not in self._hidden_imports(args)
