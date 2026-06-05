"""
api.routes.stream
==================
WebSocket streaming endpoint for real-time token delivery.
"""
from __future__ import annotations

import asyncio
import os
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.agent_graph import build_initial_state

router = APIRouter()


@router.websocket("/stream")
async def stream_chat(
    websocket: WebSocket,
    api_key: str = None,
):
    """
    WebSocket endpoint for streaming ARIA responses.

    Protocol:
    - Client sends JSON: {"message": "...", "session_id": "...", "user_id": "..."}
    - Server sends:
        {"type": "plan_step", "data": {"action": "..."}}
        {"type": "token",     "data": {"token": "word "}} (repeated)
        {"type": "done",      "data": {citations, model_used, latency_ms, confidence}}
    - On error:
        {"type": "error", "data": {"message": "..."}}
    """
    await websocket.accept()

    # Auth check
    expected = os.getenv("ARIA_API_KEY", "")
    if expected and api_key != expected:
        await websocket.close(code=1008, reason="Unauthorized")
        return

    graph = websocket.app.state.graph

    try:
        while True:
            data = await websocket.receive_json()
            message: str = data.get("message", "")
            session_id: str = data.get(
                "session_id", f"sess_{uuid.uuid4().hex[:8]}"
            )
            user_id: str = data.get("user_id", "default_user")

            if not message.strip():
                await websocket.send_json(
                    {"type": "error", "data": {"message": "Empty message."}}
                )
                continue

            # Acknowledge receipt with a plan_step event
            await websocket.send_json(
                {"type": "plan_step", "data": {"action": "Processing your request…"}}
            )

            # Run the full agent graph
            state = build_initial_state(message, session_id, user_id)
            result_state = await graph.ainvoke(state)

            response: str = result_state.get("final_response") or ""

            # Stream response word by word
            words = response.split(" ")
            for i, word in enumerate(words):
                await websocket.send_json(
                    {"type": "token", "data": {"token": word + " "}}
                )
                if i % 10 == 0:
                    await asyncio.sleep(0.01)

            # Serialise citations to plain dicts
            citations_raw = result_state.get("citations") or []
            try:
                citations = [c.model_dump() for c in citations_raw]
            except Exception:
                citations = [str(c) for c in citations_raw]

            await websocket.send_json(
                {
                    "type": "done",
                    "data": {
                        "citations": citations,
                        "model_used": result_state.get("model_used", ""),
                        "latency_ms": result_state.get("total_latency_ms", 0),
                        "confidence": result_state.get("confidence_indicator", "medium"),
                    },
                }
            )

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json(
                {"type": "error", "data": {"message": str(exc)}}
            )
            await websocket.close()
        except Exception:
            pass
