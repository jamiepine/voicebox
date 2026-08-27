"""
backend/routes/captures.py's capture_readiness_endpoint looked up the LLM
config by `c.model_size == saved.llm_model` — that only worked because
model_size ("0.6B") and the persisted value happened to be the same shape
when qwen_llm was the only engine. Now that saved.llm_model is a model_name
("minicpm5-1b"), the lookup must key off model_name instead, or MiniCPM5-1B
(whose model_size is "1B", not "minicpm5-1b") would never resolve.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")

from backend.routes import captures as captures_routes


@pytest.mark.asyncio
async def test_readiness_resolves_minicpm_engine_by_model_name(monkeypatch):
    fake_settings = MagicMock()
    fake_settings.stt_model = "turbo"
    fake_settings.llm_model = "minicpm5-1b"
    monkeypatch.setattr(
        captures_routes.settings_service, "get_capture_settings", lambda db: fake_settings
    )
    monkeypatch.setattr(captures_routes, "is_model_cached", lambda repo: True)

    response = await captures_routes.capture_readiness_endpoint(db=MagicMock())

    assert response.llm.model_name == "minicpm5-1b"
    assert response.llm.size == "1B"
