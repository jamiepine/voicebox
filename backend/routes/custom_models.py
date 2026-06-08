"""Custom HuggingFace model management endpoints."""

import logging

from fastapi import APIRouter, HTTPException

from .. import custom_models, models

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/custom-models")
async def list_custom_models():
    """List all user-added custom HuggingFace models."""
    try:
        entries = custom_models.list_custom_models()
        return [models.CustomModelResponse(**e) for e in entries]
    except Exception as e:
        logger.error("Failed to list custom models", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/custom-models", status_code=201)
async def add_custom_model(request: models.CustomModelCreate):
    """Add a new custom HuggingFace model."""
    try:
        entry = custom_models.add_custom_model(
            hf_repo_id=request.hf_repo_id,
            display_name=request.display_name,
            engine=request.engine,
        )
        return models.CustomModelResponse(**entry)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Failed to add custom model", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/custom-models/{model_id}")
async def get_custom_model(model_id: str):
    """Get a specific custom model by ID."""
    entry = custom_models.get_custom_model(model_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Custom model '{model_id}' not found")
    return models.CustomModelResponse(**entry)


@router.delete("/custom-models/{model_id}")
async def delete_custom_model(model_id: str):
    """Remove a custom model definition and delete its cached files if present."""
    import shutil
    from pathlib import Path

    from huggingface_hub import constants as hf_constants

    cm = custom_models.get_custom_model(model_id)
    if not cm:
        raise HTTPException(status_code=404, detail=f"Custom model '{model_id}' not found")

    # Get HF repo ID before deleting the definition
    hf_repo_id = cm.get("hf_repo_id", "")

    deleted = custom_models.delete_custom_model(model_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Custom model '{model_id}' not found")

    # Try to delete the cache files on disk
    if hf_repo_id:
        try:
            cache_dir = hf_constants.HF_HUB_CACHE
            repo_cache_dir = Path(cache_dir) / ("models--" + hf_repo_id.replace("/", "--"))
            if repo_cache_dir.exists():
                shutil.rmtree(repo_cache_dir)
        except Exception as e:
            logger.warning(f"Failed to delete cache dir for custom model '{model_id}': {e!s}")

    return {"message": f"Custom model '{model_id}' and its cache removed"}
