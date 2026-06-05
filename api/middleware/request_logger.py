"""
api.middleware.request_logger
==============================
Starlette middleware that logs every HTTP request as a structured JSON line
and attaches an X-Request-ID header to every response.
"""
from __future__ import annotations

import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("aria.http")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log every HTTP request/response as a compact JSON event.

    Also attaches a unique ``X-Request-ID`` header to every response and
    stores it at ``request.state.request_id`` for use inside route handlers.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.monotonic()
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        request.state.request_id = request_id

        response: Response = await call_next(request)

        latency_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": latency_ms,
                },
                ensure_ascii=False,
            )
        )
        response.headers["X-Request-ID"] = request_id
        return response
