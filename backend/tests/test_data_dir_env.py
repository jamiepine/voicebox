"""
Unit tests for the ``VOICEBOX_DATA_DIR`` environment variable.

``backend/README.md`` documents ``VOICEBOX_DATA_DIR`` alongside ``--data-dir``,
but the default was previously hardcoded to ``./data``, so a bare
``uvicorn backend.main:app`` (how ``just dev`` starts the backend) always wrote
to the repo instead of the app data dir.

The variable is read at import time, so each test reloads the module under a
patched environment.

NOTE: These tests reload ``config``, which rebinds its module-global
``_data_dir``. Import the module fresh rather than holding a reference across
reloads.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402


@pytest.fixture
def reload_config(monkeypatch):
    """Reload ``config`` with a patched environment, restoring it afterwards."""

    def _reload(data_dir: str | None):
        if data_dir is None:
            monkeypatch.delenv("VOICEBOX_DATA_DIR", raising=False)
        else:
            monkeypatch.setenv("VOICEBOX_DATA_DIR", data_dir)
        return importlib.reload(config)

    yield _reload
    # Leave the module matching the real process environment for later tests.
    importlib.reload(config)


def test_env_var_sets_data_dir(reload_config, tmp_path):
    target = tmp_path / "appdata"
    cfg = reload_config(str(target))

    assert cfg.get_data_dir() == target.resolve()


def test_defaults_to_local_data_dir_when_unset(reload_config):
    cfg = reload_config(None)

    assert cfg.get_data_dir() == Path("data").resolve()


def test_empty_env_var_falls_back_to_default(reload_config):
    cfg = reload_config("")

    assert cfg.get_data_dir() == Path("data").resolve()


def test_relative_env_var_is_resolved_to_absolute(reload_config):
    cfg = reload_config("relative/data")

    assert cfg.get_data_dir().is_absolute()
    assert cfg.get_data_dir() == Path("relative/data").resolve()


def test_set_data_dir_still_wins_over_env_var(reload_config, tmp_path):
    """``--data-dir`` calls set_data_dir() after import, so it must take
    precedence over the environment variable."""
    cfg = reload_config(str(tmp_path / "from-env"))
    explicit = tmp_path / "from-flag"

    cfg.set_data_dir(explicit)

    assert cfg.get_data_dir() == explicit.resolve()
