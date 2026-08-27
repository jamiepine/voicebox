"""Kokoro's preset-voice lookup must not hit huggingface.co when the base
model was loaded from a ModelScope local directory.

``kokoro.KPipeline.load_single_voice`` calls
``hf_hub_download(repo_id=self.repo_id, filename=f"voices/{voice}.pt")``
unless the voice string already ends in ``.pt`` (in which case it's treated
as a direct file path). ModelScope's Kokoro mirror ships the same
``voices/<id>.pt`` layout as the HF repo, so when the model came from a
ModelScope directory, KokoroTTSBackend must resolve the preset voice id to
that local file path before handing it to the pipeline — see
specs/001-modelscope-download-source/research.md (post-review fix,
2026-08-27).
"""

from backend.backends.kokoro_backend import KokoroTTSBackend


def test_resolves_to_local_pt_path_when_loaded_from_modelscope(tmp_path):
    backend = KokoroTTSBackend()
    local_dir = tmp_path / "AI-ModelScope--Kokoro-82M"
    voices_dir = local_dir / "voices"
    voices_dir.mkdir(parents=True)
    (voices_dir / "af_heart.pt").write_bytes(b"fake")
    backend._local_model_dir = str(local_dir)

    resolved = backend._resolve_voice("af_heart")

    assert resolved.endswith(".pt")
    assert resolved == str(voices_dir / "af_heart.pt")


def test_falls_back_to_bare_id_when_local_voice_file_missing(tmp_path):
    """A voice missing from the mirror (shouldn't normally happen, but the
    mirror isn't guaranteed complete) falls back to the bare id — same
    failure mode as today (hf_hub_download), not a new crash."""
    backend = KokoroTTSBackend()
    local_dir = tmp_path / "AI-ModelScope--Kokoro-82M"
    (local_dir / "voices").mkdir(parents=True)
    backend._local_model_dir = str(local_dir)

    resolved = backend._resolve_voice("af_heart")

    assert resolved == "af_heart"


def test_returns_bare_id_unchanged_when_not_loaded_from_modelscope():
    backend = KokoroTTSBackend()
    backend._local_model_dir = None

    resolved = backend._resolve_voice("af_heart")

    assert resolved == "af_heart"
