"""
tools.web_search
================
Perform web searches using Tavily with a DuckDuckGo fallback.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

async def web_search(
    query: str,
    max_results: int = 5,
    include_domains: Optional[List[str]] = None,
    search_depth: str = "basic"
) -> Dict[str, Any]:
    """Search the web using Tavily with fallback to DuckDuckGo."""
    api_key = os.getenv("TAVILY_API_KEY")

    if api_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            kwargs = {"max_results": max_results, "search_depth": search_depth}
            if include_domains:
                kwargs["include_domains"] = include_domains
            
            response = client.search(query, **kwargs)
            results = []
            for r in response.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                    "published_date": r.get("published_date"),
                    "relevance_score": r.get("score", 0.7)
                })
            
            return {
                "results": results,
                "provider_used": "tavily",
                "query": query,
                "total_results": len(results)
            }
        except Exception as e:
            logger.warning(f"Tavily search failed, falling back to DDG: {e}")

    # Fallback to DuckDuckGo
    try:
        from duckduckgo_search import DDGS
        ddgs = DDGS()
        raw_results = list(ddgs.text(query, max_results=max_results))
        results = []
        for r in raw_results:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
                "published_date": None,
                "relevance_score": 0.7
            })
            
        return {
            "results": results,
            "provider_used": "duckduckgo",
            "query": query,
            "total_results": len(results)
        }
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return {
            "results": [],
            "provider_used": "none",
            "query": query,
            "total_results": 0,
            "error": str(e)
        }
