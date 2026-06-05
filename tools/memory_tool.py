"""
tools.memory_tool
=================
Stub for the MemoryAgent (built in Prompt 05).
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from memory.schemas import MemoryWrite

async def memory_tool(
    operation: str, 
    user_id: str, 
    content: str = "", 
    memory_id: Optional[str] = None,
    memory_manager: Any = None
) -> Dict[str, Any]:
    """Execute memory operations, utilizing MemoryManager if provided."""
    if memory_manager:
        try:
            if operation == "read":
                results = await memory_manager.retrieve_episodic(user_id, query=content, top_k=10)
                return {"operation": "read", "memories": [m.model_dump() for m in results], "success": True}
            elif operation == "write":
                mw = MemoryWrite(
                    content=content, 
                    category="FACT", 
                    importance_score=0.5, 
                    user_id=user_id, 
                    session_id="system"
                )
                mid = await memory_manager._episodic.write(user_id, mw)
                return {"operation": "write", "memory_id": mid, "success": True}
            elif operation == "delete":
                if not memory_id:
                    return {"operation": "delete", "success": False, "error": "memory_id required"}
                success = await memory_manager.delete_memory(user_id, memory_id)
                return {"operation": "delete", "success": success}
            elif operation == "list":
                results = await memory_manager.get_all_memories(user_id)
                return {"operation": "list", "memories": [m.model_dump() for m in results], "success": True}
        except Exception as e:
            return {"operation": operation, "success": False, "error": str(e)}

    # Stub behavior if no manager provided
    if operation == "read":
        return {"operation": "read", "memories": [], "success": True}
    elif operation == "write":
        return {"operation": "write", "memory_id": "stub", "success": True}
    elif operation == "delete":
        return {"operation": "delete", "success": True}
    elif operation == "list":
        return {"operation": "list", "memories": [], "success": True}
    else:
        return {"operation": operation, "success": False, "error": f"Unknown operation: {operation}"}
