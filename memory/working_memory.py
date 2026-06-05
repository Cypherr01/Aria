"""
memory.working_memory
=====================
Manages the short-term working memory (conversation context) for a session.
Handles automatic summarization when context gets too large.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from db.repositories.session_repo import SessionRepository

logger = logging.getLogger(__name__)


class WorkingMemory:
    """Manages conversational context and automatic summarization."""

    def __init__(
        self,
        session_id: str,
        db_path: str,
        token_threshold: float = 0.80,
        summarize_turns: int = 10,
    ):
        self.session_id = session_id
        self.db_path = db_path
        self.token_threshold = token_threshold
        self.summarize_turns = summarize_turns
        self._turns: List[dict] = []  # [{role, content, tokens}]
        self._repo = SessionRepository(db_path)

    async def load_from_db(self) -> None:
        """Load all non-summarized turns from the DB into the cache."""
        turns = await self._repo.get_turns(self.session_id)
        # turns are dicts: {"role": x, "content": y, "tokens": z}
        self._turns = []
        for t in turns:
            self._turns.append({
                "id": t.get("id", None),
                "role": t["role"],
                "content": t["content"],
                "tokens": t.get("tokens", int(len(t["content"].split()) * 1.3))
            })

    async def add_turn(self, role: str, content: str, model_used: Optional[str] = None) -> None:
        """Add a new turn to the DB and working cache."""
        tokens = int(len(content.split()) * 1.3)
        turn_id = await self._repo.save_turn(
            self.session_id, role, content, tokens, model_used
        )
        self._turns.append({
            "id": turn_id,
            "role": role,
            "content": content,
            "tokens": tokens
        })

    def get_context_messages(self) -> List[dict]:
        """Return the current context ready for LLM prompt injection."""
        return [{"role": t["role"], "content": t["content"]} for t in self._turns]

    def estimate_total_tokens(self) -> int:
        """Sum the estimated tokens in current working memory."""
        return sum(t.get("tokens", 0) for t in self._turns)

    async def maybe_summarize(self, llm: Optional[Any] = None) -> bool:
        """
        Summarize older turns if working memory exceeds the threshold.
        If llm is None, drops the oldest turns instead.
        """
        threshold_tokens = self.token_threshold * 4096
        if self.estimate_total_tokens() < threshold_tokens:
            return False

        if len(self._turns) <= self.summarize_turns:
            return False  # Not enough turns to summarize meaningfully

        oldest_turns = self._turns[:self.summarize_turns]
        remaining_turns = self._turns[self.summarize_turns:]

        # Extract IDs to mark as summarized
        turn_ids = [t["id"] for t in oldest_turns if t.get("id") is not None]

        if llm is None:
            # Fallback: just drop the turns
            logger.warning(
                "Working memory exceeded threshold but no LLM provided. Dropping oldest %d turns.",
                len(oldest_turns)
            )
            self._turns = remaining_turns
            if turn_ids:
                await self._repo.mark_turns_summarized(self.session_id, turn_ids)
            return True

        # Construct prompt for LLM
        convo_text = "\n".join([f"{t['role'].capitalize()}: {t['content']}" for t in oldest_turns])
        prompt = (
            f"Summarize this conversation segment in 2-3 sentences.\n\n"
            f"Conversation:\n{convo_text}"
        )

        try:
            from langchain_core.messages import HumanMessage
            from models.router import Tier
            messages = [HumanMessage(content=prompt)]
            
            # Since llm here is expected to be a router
            response, _ = await llm.call_with_rotation(tier=Tier.SUMMARIZATION, messages=messages)
            summary = response.content

            if turn_ids:
                await self._repo.mark_turns_summarized(self.session_id, turn_ids)
            
            await self._repo.save_summary(self.session_id, summary)
            
            summary_tokens = int(len(summary.split()) * 1.3)
            summary_turn = {
                "id": None, 
                "role": "system", 
                "content": f"[Summary]: {summary}",
                "tokens": summary_tokens
            }
            
            self._turns = [summary_turn] + remaining_turns
            return True
        except Exception as e:
            logger.error(f"Failed to summarize working memory: {e}")
            return False
