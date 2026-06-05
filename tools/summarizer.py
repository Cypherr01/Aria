"""
tools.summarizer
================
Summarize content or URLs using Trafilatura and an LLM Router.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from models.router import Tier

logger = logging.getLogger(__name__)

async def summarizer(
    content: Optional[str] = None, 
    url: Optional[str] = None,
    target_length: int = 200,
    router: Any = None
) -> Dict[str, Any]:
    """Summarize content using an LLM router, fetching from URL if needed."""
    if url and not content:
        try:
            import trafilatura
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                extracted = trafilatura.extract(downloaded)
                if extracted:
                    content = extracted
                else:
                    return {"error": "Could not extract text from URL"}
            else:
                return {"error": "Could not fetch URL"}
        except Exception as e:
            return {"error": f"URL fetching failed: {str(e)}"}

    if not content:
        return {"error": "No content or valid URL provided"}

    word_count_original = len(content.split())

    if router:
        prompt = (
            f"Summarize the following in ~{target_length} words. "
            f"Extract 3-5 key points.\n"
            f"Return JSON: {{summary, key_points}}\n\n"
            f"Content:\n{content}"
        )
        try:
            # We construct a LangChain message format since call_with_rotation expects messages
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=prompt)]
            
            response, model_name = await router.call_with_rotation(tier=Tier.SUMMARIZATION, messages=messages)
            
            # Extract JSON from response
            import json
            import re
            
            text = response.content
            # Try to find JSON block
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(text)
                
            summary = data.get("summary", "")
            key_points = data.get("key_points", [])
            word_count_summary = len(summary.split())
            
            return {
                "summary": summary,
                "compression_ratio": word_count_summary / max(1, word_count_original),
                "key_points": key_points,
                "word_count_original": word_count_original,
                "word_count_summary": word_count_summary,
                "used_llm": True,
                "model_used": model_name
            }
        except Exception as e:
            logger.warning(f"Summarization via LLM failed, using crude fallback: {e}")

    # Fallback if no router or LLM fails
    words = content.split()
    summary = " ".join(words[:target_length])
    word_count_summary = len(summary.split())
    
    return {
        "summary": summary,
        "compression_ratio": word_count_summary / max(1, word_count_original),
        "key_points": ["Fallback summary used - no key points extracted"],
        "word_count_original": word_count_original,
        "word_count_summary": word_count_summary,
        "used_llm": False
    }
