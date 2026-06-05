"""
responsible_ai.pii_detector
============================
PII detection and redaction for log sanitisation.

IMPORTANT: redact_for_logging() is called ONLY on strings that go to logs.
It is NEVER called on user-facing responses.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class PIIDetector:
    """Detect and redact personally identifiable information."""

    PATTERNS: dict[str, str] = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE":       r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "IP_ADDRESS":  r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    }

    def __init__(self) -> None:
        self._compiled: dict[str, re.Pattern] = {
            name: re.compile(pattern)
            for name, pattern in self.PATTERNS.items()
        }
        self.nlp: Optional[object] = None
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            pass

    def redact_for_logging(self, text: str) -> str:
        """
        Redact PII for log entries ONLY.

        Applies regex patterns first, then spaCy NER on the first 1000
        characters. Returns the redacted string.

        .. warning::
            Do NOT call this on user-facing responses.
        """
        if not text:
            return text

        redacted = text

        # 1. Regex redaction
        for name, pattern in self._compiled.items():
            redacted = pattern.sub(f"[REDACTED:{name}]", redacted)

        # 2. NER-based redaction (PERSON, ORG) via overlapping chunks
        if self.nlp is not None:
            try:
                chunk_size = 1000
                overlap = 50
                entities_to_redact = set()

                i = 0
                while i < len(redacted):
                    end_idx = min(i + chunk_size, len(redacted))
                    chunk = redacted[i:end_idx]
                    doc = self.nlp(chunk)
                    
                    for ent in doc.ents:
                        if ent.label_ in ("PERSON", "ORG"):
                            label = "PERSON" if ent.label_ == "PERSON" else "ORG"
                            global_start = i + ent.start_char
                            global_end = i + ent.end_char
                            entities_to_redact.add((global_start, global_end, label))
                            
                    if end_idx == len(redacted):
                        break
                    i += (chunk_size - overlap)

                # Sort by start offset descending (right-to-left)
                sorted_entities = sorted(list(entities_to_redact), key=lambda x: x[0], reverse=True)
                
                last_replaced = len(redacted) + 1
                for start, end, label in sorted_entities:
                    if end > last_replaced:
                        # Skip if it overlaps with an already redacted section
                        continue
                    redacted = redacted[:start] + f"[REDACTED:{label}]" + redacted[end:]
                    last_replaced = start

            except Exception as exc:
                logger.debug("NER PII redaction failed: %s", exc)

        return redacted

    def has_pii(self, text: str) -> bool:
        """
        Return True if *text* contains any PII pattern.

        Used for content-filtering decisions, not for redaction.
        """
        for pattern in self._compiled.values():
            if pattern.search(text):
                return True
        return False
