"""
tools.deep_research
===================
Wrapper around the ResearchAgent (built in Prompt 06).
"""
from __future__ import annotations

from typing import Any, Dict

from tools.web_search import web_search

async def deep_research(query: str, max_sub_questions: int = 4, router: Any = None) -> Dict[str, Any]:
    """Execute a deep research pipeline."""
    if router:
        from agents.research_agent import ResearchAgent
        agent = ResearchAgent(router)
        result = await agent.research(query, max_sub_questions)
        return result
    else:
        # Fallback to simple web search
        from tools.web_search import web_search
        return await web_search(query, max_results=max_sub_questions * 3)
