"""REST endpoints for per-MCP-client voice binding settings.

The Settings UI uses these to let users configure distinct voices per
agent (Claude Code in Morgan, Cursor in Scarlett, ...). The ``client_id``
column is the same value the MCP client sends in ``X-Voicebox-Client-Id``
(or the stdio shim pulls from ``VOICEBOX_CLIENT_ID``).
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db
from ..database.models import MCPClientBinding


router = APIRouter()


@router.get(
    "/mcp/bindings",
    response_model=models.MCPClientBindingListResponse,
)
async def list_mcp_bindings(db: Session = Depends(get_db)):
    rows = (
        db.query(MCPClientBinding)
        .order_by(MCPClientBinding.client_id)
        .all()
    )
    return models.MCPClientBindingListResponse(
        items=[models.MCPClientBindingResponse.model_validate(r) for r in rows]
    )


@router.put(
    "/mcp/bindings",
    response_model=models.MCPClientBindingResponse,
)
async def upsert_mcp_binding(
    data: models.MCPClientBindingUpsert,
    db: Session = Depends(get_db),
):
    """Create-or-update a binding. Matches by client_id."""
    row = (
        db.query(MCPClientBinding)
        .filter(MCPClientBinding.client_id == data.client_id)
        .first()
    )
    if row is None:
        row = MCPClientBinding(client_id=data.client_id)
        db.add(row)

    row.label = data.label
    row.profile_id = data.profile_id
    row.default_engine = data.default_engine
    row.default_personality = data.default_personality
    row.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(row)
    return models.MCPClientBindingResponse.model_validate(row)


@router.delete("/mcp/bindings/{client_id}")
async def delete_mcp_binding(
    client_id: str,
    db: Session = Depends(get_db),
):
    row = (
        db.query(MCPClientBinding)
        .filter(MCPClientBinding.client_id == client_id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Binding not found")
    db.delete(row)
    db.commit()
    return {"deleted": client_id}
