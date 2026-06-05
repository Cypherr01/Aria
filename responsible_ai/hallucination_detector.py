"""
responsible_ai.hallucination_detector
======================================
Detect response claims that are not grounded in retrieved source texts.

Uses router.embed() cosine similarity to compare each factual
claim sentence against every source snippet. Claims below the configured
similarity threshold are flagged (and optionally removed).
"""
from __future__ import annotations

import logging
from typing import Any, List
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.config import get_config

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Compare response sentences against source texts via embedding similarity.

    A sentence is considered *hallucinated* when its max cosine similarity
    against all source texts is below ``config.responsible_ai.hallucination_threshold``.
    """

    def __init__(self, router: Any = None) -> None:
        cfg = get_config()
        self.router = router  # ModelRouter — used for embed()
        self.threshold: float = cfg.responsible_ai.hallucination_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    async def detect(
        self, response: str, source_texts: List[str]
    ) -> tuple[str, List[str]]:
        """
        Check factual claims in *response* against *source_texts*.

        Args:
            response:     The LLM-generated response string.
            source_texts: Retrieved source snippets used to ground the response.

        Returns:
            ``(cleaned_response, flagged_claims)`` where *flagged_claims* is a
            list of up-to-100-char claim excerpts that could not be verified.
            If *source_texts* is empty, returns the original response unchanged.
        """
        if not source_texts:
            return response, []

        # Split into candidate sentences (skip very short ones — likely not claims)
        sentences = [s.strip() for s in response.split(".") if len(s.strip()) > 20]
        if not sentences:
            return response, []

        # Encode in batch via router.embed()
        source_embeddings = await self.router.embed(source_texts)
        claim_embeddings = await self.router.embed(sentences)

        flagged: List[str] = []
        flagged_indices: set[int] = set()

        source_arr = np.array(source_embeddings)
        claim_arr = np.array(claim_embeddings)

        for i, (claim, claim_emb) in enumerate(zip(sentences, claim_arr)):
            sims = cosine_similarity([claim_emb], source_arr)[0]
            max_sim = float(np.max(sims))
            if max_sim < self.threshold:
                flagged.append(claim[:100])
                if get_config().responsible_ai.remove_hallucinated_claims:
                    flagged_indices.add(i)

        if flagged_indices:
            kept = [s for i, s in enumerate(sentences) if i not in flagged_indices]
            cleaned = ". ".join(kept)
            if flagged:
                cleaned += (
                    "\n\n*Note: Some claims could not be verified against "
                    "retrieved sources.*"
                )
        else:
            cleaned = response

        if flagged:
            logger.info(
                "Hallucination detector flagged %d claim(s): %s",
                len(flagged),
                [f[:60] for f in flagged],
            )

        return cleaned, flagged

    def extract_source_texts(self, tool_outputs: List[Any]) -> List[str]:
        """
        Pull raw source text from a list of ToolOutputRecord objects.

        Understands the output shapes of:
        - ``web_search``     → ``results[].snippet``
        - ``document_rag``  → ``chunks[].chunk_text``
        - ``deep_research`` → ``summary`` / ``key_findings[].claim``
        """
        sources: List[str] = []
        for output in tool_outputs:
            # Support both Pydantic models (ToolOutputRecord) and plain dicts
            if hasattr(output, "success"):
                if not output.success:
                    continue
                o = output.output
            elif isinstance(output, dict):
                if not output.get("success", True):
                    continue
                o = output.get("output", output)
            else:
                continue

            if not isinstance(o, dict):
                continue

            if "results" in o:          # WebSearch
                sources.extend(
                    r.get("snippet", "") for r in o["results"] if r.get("snippet")
                )
            elif "chunks" in o:         # RAG
                sources.extend(
                    c.get("chunk_text", "") for c in o["chunks"] if c.get("chunk_text")
                )
            elif "summary" in o:        # ResearchAgent
                sources.append(o["summary"])
            elif "key_findings" in o:   # Deep research
                sources.extend(
                    f.get("claim", "") for f in o.get("key_findings", [])
                )

        return [s for s in sources if s]
