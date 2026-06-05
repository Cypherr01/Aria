"""
memory.memory_manager
=====================
Facade for the complete 3-tier memory system.
Provides a unified API for the graph nodes to interact with context and facts.
"""
from __future__ import annotations

from typing import Any, Dict, List

from db.repositories.memory_repo import MemoryRepository
from db.repositories.session_repo import SessionRepository
from memory.episodic_memory import EpisodicMemoryStore
from memory.schemas import EpisodicMemory
from memory.working_memory import WorkingMemory


class MemoryManager:
    """Facade for Working and Episodic Memory operations."""

    def __init__(self, chroma_path: str, db_path: str):
        self.db_path = db_path
        self.chroma_path = chroma_path
        self._memory_repo = MemoryRepository(db_path)
        self._session_repo = SessionRepository(db_path)
        self._episodic = EpisodicMemoryStore(chroma_path, self._memory_repo)
        self._working_memories: Dict[str, WorkingMemory] = {}

    def get_working_memory(self, session_id: str) -> WorkingMemory:
        """Get or initialize the working memory cache for a session."""
        if session_id not in self._working_memories:
            self._working_memories[session_id] = WorkingMemory(session_id, self.db_path)
        return self._working_memories[session_id]

    async def initialize_session(self, session_id: str, user_id: str, model_used: str = None) -> None:
        """Create a new session in DB and warm up the working memory cache."""
        await self._session_repo.create_session(session_id, user_id, model_used)
        wm = self.get_working_memory(session_id)
        await wm.load_from_db()

    async def retrieve_episodic(
        self, user_id: str, query: str, top_k: int = 10, min_importance: float = 0.15
    ) -> List[EpisodicMemory]:
        """Search long-term episodic memory for facts."""
        return await self._episodic.retrieve(user_id, query, top_k, min_importance)

    async def write_turn_to_episodic(
        self, user_id: str, session_id: str, user_message: str, assistant_response: str, router: Any
    ) -> List[str]:
        """Automatically extract and persist semantic info from a turn."""
        return await self._episodic.extract_and_write_from_turn(
            user_id, session_id, user_message, assistant_response, router
        )

    async def save_session_turn(
        self, session_id: str, user_message: str, assistant_response: str, model_used: str
    ) -> None:
        """Save a standard interaction turn to working memory."""
        wm = self.get_working_memory(session_id)
        await wm.add_turn("user", user_message)
        await wm.add_turn("assistant", assistant_response, model_used)

    async def get_conversation_context(self, session_id: str) -> List[dict]:
        """Retrieve the active sliding window of conversation context."""
        wm = self.get_working_memory(session_id)
        return wm.get_context_messages()

    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """Delete an episodic memory."""
        return await self._episodic.delete(user_id, memory_id)

    async def get_all_memories(
        self, user_id: str, min_importance: float = 0.0, limit: int = 50
    ) -> List[EpisodicMemory]:
        """Fetch the most important memories without semantic matching."""
        return await self._episodic.get_all(user_id, min_importance, limit)
