"""LLM inference endpoints."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from .. import models
from ..backends import get_llm_model_configs
from ..services import llm
from ..services.task_queue import create_background_task
from ..utils.tasks import get_task_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/llm/generate", response_model=models.LLMGenerateResponse)
async def llm_generate(request: models.LLMGenerateRequest):
    """Run a single-turn Qwen3 completion."""
    backend = llm.get_llm_model()
    model_size = request.model_size or backend.model_size

    valid_sizes = {cfg.model_size for cfg in get_llm_model_configs()}
    if model_size not in valid_sizes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid LLM size '{model_size}'. Must be one of: {sorted(valid_sizes)}",
        )

    already_loaded = backend.is_loaded() and backend.model_size == model_size
    if not already_loaded and not backend._is_model_cached(model_size):
        progress_model_name = f"qwen3-{model_size.lower()}"
        task_manager = get_task_manager()

        async def download_llm_background():
            try:
                await backend.load_model(model_size)
                task_manager.complete_download(progress_model_name)
            except Exception as e:
                task_manager.error_download(progress_model_name, str(e))

        task_manager.start_download(progress_model_name)
        create_background_task(download_llm_background())

        return JSONResponse(
            status_code=202,
            content={
                "message": f"Qwen3 {model_size} is being downloaded. Please wait and try again.",
                "model_name": progress_model_name,
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
            model_size=model_size,
            examples=examples,
        )
        return models.LLMGenerateResponse(text=text, model_size=model_size)
    except Exception as e:
        # The backend exception text can include filesystem paths and stack
        # frames — log it server-side and hand the client a generic message.
        logger.exception("LLM generate failed")
        raise HTTPException(status_code=500, detail="LLM generation failed") from e
