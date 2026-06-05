"""
tools.structured_output
=======================
Format content into specific structured formats using an LLM Router.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from models.router import Tier

logger = logging.getLogger(__name__)

async def structured_output(
    content: str, 
    format_type: str,
    router: Any = None
) -> Dict[str, Any]:
    """Convert content to a specific format."""
    valid_formats = {"json", "table", "bullets", "code_block", "numbered_list"}
    if format_type not in valid_formats:
        return {"error": f"Invalid format_type. Must be one of {valid_formats}"}

    if router:
        prompt = (
            f"Convert this content to {format_type} format. "
            f"Return ONLY the formatted content, without conversational filler.\n\n"
            f"Content:\n{content}"
        )
        try:
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=prompt)]
            
            response, _ = await router.call_with_rotation(tier=Tier.SPEED, messages=messages)
            formatted_content = response.content
            
            return {
                "formatted_content": formatted_content,
                "format_type": format_type,
                "used_llm": True
            }
        except Exception as e:
            logger.warning(f"Structured output via LLM failed, using fallback: {e}")

    # Fallback
    formatted_content = content
    if format_type == "json":
        formatted_content = json.dumps({"content": content})
    elif format_type == "table":
        # Cannot convert to table without LLM
        formatted_content = content 
    elif format_type == "bullets":
        sentences = [s.strip() for s in content.split(". ") if s.strip()]
        formatted_content = "\n".join([f"- {s}" for s in sentences])
    elif format_type == "code_block":
        formatted_content = f"```\n{content}\n```"
    elif format_type == "numbered_list":
        sentences = [s.strip() for s in content.split(". ") if s.strip()]
        formatted_content = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)])

    return {
        "formatted_content": formatted_content,
        "format_type": format_type,
        "used_llm": False
    }
