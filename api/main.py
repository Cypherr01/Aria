"""
api.main
=========
ARIA FastAPI application — entry point for uvicorn.

Start the server::

    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

The lifespan handler bootstraps all components once at startup and tears
them down cleanly on shutdown. All components are stored on ``app.state``
and accessed in routes via the dependency functions in ``api.deps``.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.background_tasks import run_decay_loop, run_maintenance_loop
from api.middleware.auth import verify_api_key
from api.middleware.request_logger import RequestLoggingMiddleware
from api.routes.analytics import router as analytics_router
from api.routes.chat import router as chat_router
from api.routes.documents import router as docs_router
from api.routes.memory import router as memory_router
from api.routes.models import router as models_router
from api.routes.stream import router as stream_router
from config.config import get_config
from core.agent_graph import build_aria_components, build_aria_graph
from db.database import init_db
from db.repositories.memory_repo import MemoryRepository
from memory.memory_decay import MemoryDecay
from observability.observability import Observability

logger = logging.getLogger("aria.main")

# Global flag indicating if the safety layer is running without NER PII redaction
RESPONSIBLE_AI_DEGRADED = False

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    STARTUP:
      1. Initialise the SQLite database schema.
      2. Instantiate all ARIA singleton components.
      3. Compile the LangGraph agent graph.
      4. Configure observability (logging + LangSmith).
      5. Store everything on ``app.state`` for route access.
      6. Start background maintenance loops (memory decay + analytics purge).

    SHUTDOWN:
      Cancel and await background tasks, then allow uvicorn to exit cleanly.
    """
    config = get_config()
    db_path = os.getenv("ARIA_DB_PATH", "./data/aria.db")
    chroma_path = os.getenv("ARIA_CHROMA_PATH", "./data/chroma")

    # Ensure data directories exist
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    os.makedirs(chroma_path, exist_ok=True)

    # Initialise database
    await init_db(db_path)

    # Check spaCy loadability via responsible_ai
    from responsible_ai.responsible_ai import RESPONSIBLE_AI_DEGRADED as RAI_DEGRADED
    global RESPONSIBLE_AI_DEGRADED
    if RAI_DEGRADED:
        logger.error(
            "spaCy en_core_web_sm not found. PII NER is disabled. "
            "Responsible AI layer is DEGRADED."
        )
        RESPONSIBLE_AI_DEGRADED = True

    # Build all singleton components
    components = build_aria_components(db_path=db_path, chroma_path=chroma_path)

    # Hydrate today's token usage from DB
    try:
        await components["router"]._load_today_token_usage()
    except Exception as exc:
        logger.warning("Could not pre-load token usage: %s", exc)

    # Compile agent graph
    graph = build_aria_graph(components)

    # Observability setup (idempotent — components factory already creates one,
    # but we call setup() here to configure logging + LangSmith env vars)
    obs: Observability = components.get("observability") or Observability(db_path=db_path)
    obs.setup(
        log_level=config.observability.log_level,
        enable_langsmith=config.observability.langsmith_enabled,
    )

    # Memory decay — owns a dedicated MemoryRepository so it doesn't share
    # connection state with the graph's MemoryManager.
    memory_repo = MemoryRepository(db_path=db_path)
    memory_decay = MemoryDecay(
        memory_repo=memory_repo,
        chroma_path=chroma_path,
    )

    # Store on app.state for dependency injection
    app.state.components = components
    app.state.graph = graph
    app.state.observability = obs
    app.state.config = config

    # ── Start background maintenance loops ────────────────────────────────────
    decay_task = asyncio.create_task(
        run_decay_loop(memory_decay),
        name="aria-memory-decay",
    )
    maintenance_task = asyncio.create_task(
        run_maintenance_loop(obs),
        name="aria-obs-maintenance",
    )
    # Store task handles so tests and health routes can inspect them
    app.state.background_tasks = [decay_task, maintenance_task]

    logger.info("ARIA started successfully — graph compiled, DB ready, background tasks running.")
    yield

    # ── SHUTDOWN — cancel background tasks gracefully ─────────────────────────
    logger.info("ARIA shutting down — cancelling background tasks.")
    for task in app.state.background_tasks:
        task.cancel()
    # Await so uvicorn doesn't exit with pending asyncio warnings
    await asyncio.gather(*app.state.background_tasks, return_exceptions=True)
    logger.info("ARIA shut down gracefully.")



# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="ARIA API",
    description=(
        "Agentic Reasoning and Intelligent Assistant — "
        "a multi-model AI system with RAG, memory, and tool execution."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

# Protected routes
app.include_router(
    chat_router,
    prefix="/chat",
    tags=["chat"],
)
app.include_router(
    docs_router,
    # documents.py already has prefix="/documents" baked in — keep as-is
    tags=["documents"],
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    memory_router,
    prefix="/memory",
    tags=["memory"],
)
app.include_router(
    analytics_router,
    prefix="/analytics",
    tags=["analytics"],
)

# Public routes (no auth)
app.include_router(
    models_router,
    prefix="/model-status",
    tags=["models"],
)
app.include_router(stream_router, tags=["streaming"])


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
async def health():
    """Public health probe — returns 200 when the application is running."""
    return {
        "status": "ok", 
        "version": "1.0.0",
        "responsible_ai_degraded": RESPONSIBLE_AI_DEGRADED
    }
