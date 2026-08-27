"""
_migrate_llm_model_names must rewrite historical bare-size llm_model values
("0.6B"/"1.7B"/"4B") to their model_name form ("qwen3-0.6b" etc) in both
capture_settings and captures, idempotently, without touching NULL rows
(a capture that never had refinement run on it has llm_model = NULL) or
values that are already in the new form.

This is what makes User Story 3 (existing installs keep working across the
upgrade) actually true — without it, a pre-existing install's saved
llm_model setting and every past capture's model attribution would stop
resolving to a real ModelConfig the moment the identifier scheme changed.
"""

import uuid

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

pytest.importorskip("torch")

from backend.database import Base, Capture, CaptureSettings
from backend.database.migrations import _migrate_llm_model_names


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


def _seed(engine, *, capture_settings_llm_model: str, capture_llm_models: list[str | None]):
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        db.add(CaptureSettings(id=1, llm_model=capture_settings_llm_model))
        for value in capture_llm_models:
            db.add(
                Capture(
                    id=str(uuid.uuid4()),
                    audio_path="x.wav",
                    source="dictation",
                    transcript_raw="hello",
                    llm_model=value,
                )
            )
        db.commit()
    finally:
        db.close()


def _read_llm_models(engine) -> tuple[list[str], list[str | None]]:
    # ORDER BY rowid: captures.id is a random UUID, so without an explicit
    # order SQLite is free to return rows in any order, and comparing to a
    # fixed-order list would be flaky.
    with engine.connect() as conn:
        settings_values = [
            row[0]
            for row in conn.execute(text("SELECT llm_model FROM capture_settings ORDER BY rowid"))
        ]
        capture_values = [
            row[0] for row in conn.execute(text("SELECT llm_model FROM captures ORDER BY rowid"))
        ]
    return settings_values, capture_values


def test_rewrites_legacy_bare_sizes_to_model_names(engine):
    _seed(engine, capture_settings_llm_model="1.7B", capture_llm_models=["0.6B", "4B", None])

    tables = set(inspect(engine).get_table_names())
    _migrate_llm_model_names(engine, tables)

    settings_values, capture_values = _read_llm_models(engine)
    assert settings_values == ["qwen3-1.7b"]
    assert capture_values == ["qwen3-0.6b", "qwen3-4b", None]


def test_leaves_already_migrated_values_alone(engine):
    _seed(engine, capture_settings_llm_model="minicpm5-1b", capture_llm_models=["qwen3-0.6b"])

    tables = set(inspect(engine).get_table_names())
    _migrate_llm_model_names(engine, tables)

    settings_values, capture_values = _read_llm_models(engine)
    assert settings_values == ["minicpm5-1b"]
    assert capture_values == ["qwen3-0.6b"]


def test_is_idempotent_on_a_second_run(engine):
    _seed(engine, capture_settings_llm_model="0.6B", capture_llm_models=["1.7B"])
    tables = set(inspect(engine).get_table_names())

    _migrate_llm_model_names(engine, tables)
    _migrate_llm_model_names(engine, tables)

    settings_values, capture_values = _read_llm_models(engine)
    assert settings_values == ["qwen3-0.6b"]
    assert capture_values == ["qwen3-1.7b"]


def test_noop_when_tables_are_absent(engine):
    """Safety net for a fresh install where run_migrations sees no tables yet."""
    _migrate_llm_model_names(engine, tables=set())
