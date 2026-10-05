"""Conversation mode — BYO-LLM voice agent loop."""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models
from ..backends import engine_has_model_sizes
from ..database import VoiceProfile as DBVoiceProfile, get_db
from ..services import history, profiles
from ..services import settings as settings_service
from ..services.generation import run_generation
from ..services.task_queue import enqueue_generation
from ..utils.tasks import get_task_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversation", tags=["conversation"])


@router.post("/turn", response_model=models.ConversationTurnResponse)
async def conversation_turn(
    request: models.ConversationTurnRequest,
    db: Session = Depends(get_db),
):
    """Run one conversational turn: user_message → LLM reply → TTS generation.

    The TTS job is enqueued in the background (same as POST /generate). The
    response includes the generation_id so the frontend can poll the SSE
    endpoint ``/generate/{id}/status`` for completion and then play the audio.
    """
    conv_settings = settings_service.get_conversation_settings(db)

    if not conv_settings.enabled:
        raise HTTPException(
            status_code=400,
            detail="Conversation mode is disabled. Enable it in Settings → Conversation.",
        )

    if not conv_settings.llm_endpoint:
        raise HTTPException(status_code=400, detail="No LLM endpoint configured.")

    # Load voice profile for personality and TTS
    profile = await profiles.get_profile(request.profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    # Build system prompt from conversation prefix + profile personality
    system_parts = []
    if conv_settings.system_prompt_prefix:
        system_parts.append(conv_settings.system_prompt_prefix)
    if getattr(profile, "personality", None):
        system_parts.append(profile.personality)
    if not system_parts:
        system_parts.append("You are a helpful AI assistant. Reply concisely.")
    system_prompt = "\n\n".join(system_parts)

    # Call the external LLM
    from ..backends.openai_compat_backend import OpenAICompatLLMBackend

    backend = OpenAICompatLLMBackend(
        endpoint=conv_settings.llm_endpoint,
        api_key=conv_settings.llm_api_key or "",
        model=conv_settings.llm_model or "llama3",
    )

    # Build the full message history including the new user turn
    history_messages = [{"role": m.role, "content": m.content} for m in request.history]
    history_messages.append({"role": "user", "content": request.user_message})

    try:
        assistant_text = backend.generate_with_history(
            history=history_messages,
            system=system_prompt,
            max_tokens=512,
            temperature=0.7,
        )
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM request failed: {str(e)}")

    if not assistant_text:
        raise HTTPException(status_code=502, detail="LLM returned an empty reply.")

    # Enqueue TTS generation for the assistant reply (same pipeline as POST /generate)
    generation_id = str(uuid.uuid4())
    engine = request.engine or "qwen"
    model_size = (request.model_size or "1.7B") if engine_has_model_sizes(engine) else None

    try:
        profiles.validate_profile_engine(profile, engine)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    task_manager = get_task_manager()

    await history.create_generation(
        profile_id=request.profile_id,
        text=assistant_text,
        language=request.language,
        audio_path="",
        duration=0,
        seed=None,
        db=db,
        generation_id=generation_id,
        status="generating",
        engine=engine,
        model_size=model_size,
        source="conversation",
    )

    task_manager.start_generation(
        task_id=generation_id,
        profile_id=request.profile_id,
        text=assistant_text,
    )

    enqueue_generation(
        generation_id,
        run_generation(
            generation_id=generation_id,
            profile_id=request.profile_id,
            text=assistant_text,
            language=request.language,
            engine=engine,
            model_size=model_size,
            seed=None,
            normalize=True,
            effects_chain=None,
            instruct=None,
            mode="generate",
        ),
    )

    audio_url = f"/generate/{generation_id}/status"

    return models.ConversationTurnResponse(
        assistant_text=assistant_text,
        generation_id=generation_id,
        audio_url=audio_url,
    )
