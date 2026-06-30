"""Security tests for model cache migration path handling."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi import HTTPException

from backend import config
from backend.routes import models as model_routes


@pytest.fixture
def owned_models_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config, "_data_dir", tmp_path / "data")
    return config.get_models_dir()


def _source_cache(tmp_path: Path) -> Path:
    source = tmp_path / "hf-cache"
    source.mkdir()
    return source


def _assert_rejected(raw_destination: str, source: Path) -> HTTPException:
    with pytest.raises(HTTPException) as exc_info:
        model_routes._resolve_migration_destination(raw_destination, source)
    assert exc_info.value.status_code == 400
    return exc_info.value


def test_migration_destination_allows_owned_root(owned_models_root: Path, tmp_path: Path) -> None:
    source = _source_cache(tmp_path)

    destination = model_routes._resolve_migration_destination(str(owned_models_root), source)

    assert destination == owned_models_root.resolve()


def test_migration_destination_allows_owned_direct_child(owned_models_root: Path, tmp_path: Path) -> None:
    source = _source_cache(tmp_path)

    destination = model_routes._resolve_migration_destination("local-cache", source)

    assert destination == owned_models_root.resolve() / "local-cache"


def test_migration_destination_rejects_absolute_outside_path(owned_models_root: Path, tmp_path: Path) -> None:
    source = _source_cache(tmp_path)
    outside = owned_models_root.parent / "outside"
    outside.mkdir()

    error = _assert_rejected(str(outside), source)

    assert "inside the Voicebox-owned models directory" in error.detail


def test_migration_destination_rejects_relative_traversal(owned_models_root: Path, tmp_path: Path) -> None:
    source = _source_cache(tmp_path)

    error = _assert_rejected("../outside", source)

    assert "inside the Voicebox-owned models directory" in error.detail


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support required")
def test_migration_destination_rejects_symlink_escape(owned_models_root: Path, tmp_path: Path) -> None:
    source = _source_cache(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = owned_models_root / "linked-cache"
    link.symlink_to(outside, target_is_directory=True)

    error = _assert_rejected(str(link), source)

    assert "cannot be a symlink" in error.detail


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support required")
def test_migration_destination_rejects_symlink_owned_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (data_dir / "models").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(config, "_data_dir", data_dir)
    source = _source_cache(tmp_path)

    error = _assert_rejected("local-cache", source)

    assert "models directory cannot be a symlink" in error.detail


def test_migration_destination_rejects_source_contained_path(owned_models_root: Path) -> None:
    source = owned_models_root

    error = _assert_rejected("nested-cache", source)

    assert "inside the current cache directory" in error.detail


def test_migration_rejects_existing_destination_model_dir(owned_models_root: Path, tmp_path: Path) -> None:
    source = _source_cache(tmp_path)
    model_dir = source / "models--voicebox--example"
    model_dir.mkdir()
    destination = owned_models_root / "target-cache"
    destination.mkdir()
    (destination / model_dir.name).mkdir()

    with pytest.raises(HTTPException) as exc_info:
        model_routes._validate_no_destination_collisions([model_dir], destination)

    assert exc_info.value.status_code == 409
    assert model_dir.name in exc_info.value.detail
