"""
observability
=============
ARIA's observability layer — structured logging, LangSmith tracing, analytics.

Exposes:
  - Observability       — application-level facade (setup / get_tracer / analytics)
  - StructuredLogger    — JSON event emitter with PII redaction
  - ARIALangSmithTracer — LangChain callback handler for LLM/tool tracing
"""
from __future__ import annotations

try:
    from observability.structured_logger import StructuredLogger
    from observability.langsmith_tracer import ARIALangSmithTracer
    from observability.observability import Observability
    __all__ = ["Observability", "StructuredLogger", "ARIALangSmithTracer"]
except ImportError:
    __all__ = []
