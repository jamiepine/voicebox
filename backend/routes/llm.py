"""LLM inference endpoints."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from .. import models
from ..backends import get_model_config
from ..services import llm
from ..services.task_queue import create_background_task
from ..utils.tasks import get_task_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/llm/generate", response_model=models.LLMGenerateResponse)
async def llm_generate(request: models.LLMGenerateRequest):
    """Run a single-turn LLM completion (Qwen3 or MiniCPM5, whichever model_size names)."""
    # Despite the field name, request.model_size holds a model_name — resolve
    # it to the (engine, bare size) pair the backend classes actually expect.
    # LLMGenerateRequest's field validator already rejects unknown model
    # names at the request-parsing boundary, so get_model_config is
    # guaranteed to find a match here.
    model_name = request.model_size or "qwen3-0.6b"
    config = get_model_config(model_name)

    backend = llm.get_llm_model(config.engine)
    resolved_size = config.model_size

    already_loaded = backend.is_loaded() and backend.model_size == resolved_size
    if not already_loaded and not backend._is_model_cached(resolved_size):
        task_manager = get_task_manager()

        async def download_llm_background():
            try:
                await backend.load_model(resolved_size)
                task_manager.complete_download(model_name)
            except Exception as e:
                task_manager.error_download(model_name, str(e))

        task_manager.start_download(model_name)
        create_background_task(download_llm_background())

        return JSONResponse(
            status_code=202,
            content={
                "message": f"{config.display_name} is being downloaded. Please wait and try again.",
                "model_name": model_name,
                "downloading": True,
            },
        )

    examples: list[tuple[str, str]] | None = None
    if request.examples:
        for pair in request.examples:
            if len(pair) != 2:
                raise HTTPException(
                    status_code=400,
                    detail="Each example must be a [user, assistant] pair",
                )
        examples = [(pair[0], pair[1]) for pair in request.examples]

    try:
        text = await backend.generate(
            prompt=request.prompt,
            system=request.system,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            model_size=resolved_size,
            examples=examples,
        )
        return models.LLMGenerateResponse(text=text, model_size=model_name)
    except Exception as e:
        # The backend exception text can include filesystem paths and stack
        # frames — log it server-side and hand the client a generic message.
        logger.exception("LLM generate failed")
        raise HTTPException(status_code=500, detail="LLM generation failed") from e
