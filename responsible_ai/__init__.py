"""
responsible_ai
==============
ARIA's responsible-AI safety layer.

Exposes:
  - ResponsibleAI     — single pipeline entrypoint (check_input / process_output)
  - PIIDetector       — regex + NER PII redaction (logs only)
  - InjectionGuard    — prompt-injection detection + unicode sanitisation
  - HallucinationDetector — embedding-based claim grounding check
  - ContentFilter     — LLM-backed harm classification
"""
from __future__ import annotations

from responsible_ai.pii_detector import PIIDetector
from responsible_ai.injection_guard import InjectionGuard
from responsible_ai.hallucination_detector import HallucinationDetector
from responsible_ai.content_filter import ContentFilter
from responsible_ai.responsible_ai import ResponsibleAI

__all__ = [
    "ResponsibleAI",
    "PIIDetector",
    "InjectionGuard",
    "HallucinationDetector",
    "ContentFilter",
]
