"""
api.routes.chat
================
Chat, session history, and session management endpoints.
"""
from __future__ import annotations

import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.deps import get_graph, get_memory_manager, get_observability, get_responsible_ai
from api.middleware.auth import verify_api_key
from api.schemas.chat_schemas import ChatRequest, ChatResponse, PlanTraceStep
from core.agent_graph import build_initial_state
from db.repositories.session_repo import SessionRepository
from config.config import get_config
import os

router = APIRouter()


def _session_repo(request: Request) -> SessionRepository:
    db_path = os.getenv("ARIA_DB_PATH", "./data/aria.db")
    return SessionRepository(db_path=db_path)


# ── POST /chat ─────────────────────────────────────────────────────────────────

@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    _api_key: str = Depends(verify_api_key),
    graph=Depends(get_graph),
    responsible_ai=Depends(get_responsible_ai),
    memory_manager=Depends(get_memory_manager),
    obs=Depends(get_observability),
):
    """
    Run the full ARIA agent graph for a user message and return the response.

    Steps:
    1. Safety check via responsible_ai.check_input().
    2. Initialise session if new.
    3. Build initial ARIAState and invoke the graph.
    4. Return ChatResponse with trace, citations, and suggestions.
    """
    request_id = getattr(request.state, "request_id", f"req_{uuid.uuid4().hex[:8]}")
    t_start = time.monotonic()

    # 1. Safety check
    is_safe, reason = await responsible_ai.check_input(
        body.message, body.session_id
    )
    if not is_safe:
        raise HTTPException(status_code=400, detail=reason or "Unsafe input detected.")

    # 2. Initialise session
    session_repo = _session_repo(request)
    try:
        existing = await session_repo.get_session(body.session_id)
        if not existing:
            await memory_manager.initialize_session(
                session_id=body.session_id,
                user_id=body.user_id,
            )
    except Exception:
        pass  # Non-fatal — continue without session row

    # 3. Build conversation history from working memory
    try:
        history = await memory_manager.get_conversation_context(body.session_id)
    except Exception:
        history = []

    # 4. Invoke graph
    initial_state = build_initial_state(
        user_message=body.message,
        session_id=body.session_id,
        user_id=body.user_id,
        conversation_history=history,
        selected_primary_model=body.primary_model,
    )
    result = await graph.ainvoke(initial_state)

    total_latency_ms = int((time.monotonic() - t_start) * 1000)

    # 5. Extract plan trace
    plan_trace = None
    if body.debug_mode:
        execution_plan = result.get("execution_plan") or []
        tool_outputs = result.get("tool_outputs") or []
        plan_trace = []
        for i, step in enumerate(execution_plan):
            latency = tool_outputs[i].latency_ms if i < len(tool_outputs) else None
            status = "success" if (i < len(tool_outputs) and tool_outputs[i].success) else "pending"
            plan_trace.append(
                PlanTraceStep(
                    step=step.step_number,
                    action=step.action,
                    status=status,
                    latency_ms=latency,
                )
            )

    # 6. Build citations list
    citations = [c.model_dump() for c in (result.get("citations") or [])]

    # 7. Log event
    obs.logger.log(
        session_id=body.session_id,
        user_id=body.user_id,
        agent="ChatRouter",
        action="chat_request",
        success=True,
        latency_ms=total_latency_ms,
        model_used=result.get("model_used", ""),
    )

    # 8. Resolve primary_model_resolved for response auditability
    cfg = get_config().model_tiers
    primary_chain = list(cfg.reasoning_chain)
    if body.primary_model and body.primary_model in primary_chain:
        primary_model_resolved = body.primary_model
    else:
        primary_model_resolved = primary_chain[0] if primary_chain else None

    return ChatResponse(
        response=result.get("final_response") or "",
        session_id=body.session_id,
        intent_type=result.get("intent_type", "conversational"),
        confidence_indicator=result.get("confidence_indicator", "medium"),
        citations=citations,
        follow_up_suggestions=result.get("follow_up_suggestions") or [],
        plan_trace=plan_trace,
        model_used=result.get("model_used", ""),
        primary_model_resolved=primary_model_resolved,
        model_chain_used="primary",
        total_latency_ms=total_latency_ms,
        request_id=request_id,
        reflection_skipped=result.get("reflection_skipped", False),
    )


# ── GET /{session_id}/history ──────────────────────────────────────────────────

@router.get("/{session_id}/history")
async def get_session_history(
    session_id: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _api_key: str = Depends(verify_api_key),
):
    """Return paginated conversation turns for a session."""
    session_repo = _session_repo(request)
    session = await session_repo.get_session(session_id)
    turns = await session_repo.get_turns(session_id, limit=limit, offset=offset)
    return {
        "session_id": session_id,
        "messages": turns,
        "total_turns": len(turns),
        "created_at": session.get("created_at") if session else None,
        "last_active": session.get("last_active") if session else None,
    }


# ── DELETE /{session_id} ───────────────────────────────────────────────────────

@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    request: Request,
    clear_history: bool = Query(default=True),
    _api_key: str = Depends(verify_api_key),
):
    """Delete a session and optionally all of its conversation turns."""
    session_repo = _session_repo(request)
    await session_repo.delete_session(session_id, clear_history=clear_history)
    return {"success": True, "session_id": session_id}
