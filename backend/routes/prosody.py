"""Prosody transformer endpoints.

Two things a caller needs before generating: what the script will actually be
turned into, and optional help writing the markup in the first place.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db
from ..services import pronunciation
from ..services.prosody import annotate, compile_plan, rules_from_entries
from ..services.prosody.ir import Silence, Speech
from ..services.prosody.llm_annotate import (
    DEFAULT_MODEL_SIZE,
    LLMUnavailableError,
    annotate_with_llm,
    is_llm_available,
)
from ..services.prosody.parser import ProsodyParseError
from ..services.prosody.pipeline import engine_capabilities

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/prosody/preview", response_model=models.ProsodyPreviewResponse)
async def preview_prosody(
    data: models.ProsodyPreviewRequest,
    db: Session = Depends(get_db),
):
    """Show what a script compiles to, without generating anything.

    The dictionary and the markup both resolve into a plan long before any
    audio exists, so this is the difference between trusting the pipeline and
    inspecting it: every cut, every language, every silence, and everything the
    chosen engine cannot honour.
    """
    entries = pronunciation.get_entries(
        db, language=data.language, profile_id=data.profile_id
    )
    annotated, applied_terms = annotate(data.text, rules_from_entries(entries))

    supports_instruct, languages = engine_capabilities(data.engine)
    try:
        plan = compile_plan(
            annotated,
            engine=data.engine,
            default_language=data.language,
            supports_instruct=supports_instruct,
            engine_languages=languages,
            base_instruct=data.instruct,
        )
    except ProsodyParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    nodes = [
        models.ProsodyPlanNode(
            kind="silence" if isinstance(n, Silence) else "speech",
            text=getattr(n, "text", None),
            language=getattr(n, "language", None),
            rate=getattr(n, "rate", None),
            instruct=getattr(n, "instruct", None),
            source_text=getattr(n, "source_text", None),
            ms=getattr(n, "ms", None),
        )
        for n in plan.nodes
    ]

    return models.ProsodyPreviewResponse(
        original=data.text,
        markup=annotated,
        dictionary_terms=applied_terms,
        nodes=nodes,
        warnings=[
            models.ProsodyPlanWarning(code=w.code, detail=w.detail) for w in plan.warnings
        ],
        run_count=sum(1 for n in plan.nodes if isinstance(n, Speech)),
        is_trivial=plan.is_trivial,
    )


@router.get("/prosody/annotate/availability")
async def annotation_availability(model_size: str = DEFAULT_MODEL_SIZE):
    """Whether LLM annotation can run right now.

    Lets a client hide or disable the action instead of offering something that
    will fail. Annotation is optional help -- everything else works without it.
    """
    return {"available": is_llm_available(model_size), "model_size": model_size}


@router.post("/prosody/annotate", response_model=models.ProsodyAnnotateResponse)
async def annotate_prosody(data: models.ProsodyAnnotateRequest):
    """Draft prosody markup for a script using the local LLM.

    The result is markup for a human to review, not audio. Nothing is stored
    and nothing is generated -- accepting the suggestion means keeping the
    returned text, which then runs through exactly the same pipeline as markup
    typed by hand.

    A suggestion whose words differ from the input is rejected: the model can
    fail to help, but it cannot rewrite the script.
    """
    try:
        result = await annotate_with_llm(
            data.text,
            language=data.language,
            model_size=data.model_size or DEFAULT_MODEL_SIZE,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return models.ProsodyAnnotateResponse(
        original=data.text,
        markup=result.markup,
        accepted=result.accepted,
        changed=result.changed,
        rejected_reason=result.rejected_reason,
        model_size=result.model_size,
        attempts=result.attempts,
    )
