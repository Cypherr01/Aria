"""
agents.research_agent
=====================
Handles complex, multi-step research queries by decomposing them,
searching in parallel, and synthesizing the results.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

from shared.constants import TaskType
from tools.web_search import web_search

logger = logging.getLogger(__name__)


class ResearchAgent:
    """Agent for deep, multi-step research tasks."""

    def __init__(self, router: Any):
        self.router = router

    async def research(
        self, query: str, max_sub_questions: int = 4, max_results_per_question: int = 5, max_sources: int = 10
    ) -> Dict[str, Any]:
        """Execute full research pipeline."""
        try:
            # Step 1: Decompose
            sub_questions = await self._decompose_query(query, max_sub_questions)
            
            # Step 2: Search in parallel
            tasks = [web_search(q, max_results=max_results_per_question) for q in sub_questions]
            results_gathered = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_results = []
            for r in results_gathered:
                if isinstance(r, dict) and "results" in r:
                    all_results.extend(r["results"])
                    
            # Step 3: Deduplicate by URL
            seen_urls = set()
            deduped = []
            for r in all_results:
                if isinstance(r, dict) and r.get("url") not in seen_urls:
                    seen_urls.add(r["url"])
                    deduped.append(r)
            
            if not deduped:
                return {
                    "summary": "No research results found.",
                    "key_findings": [],
                    "sources": [],
                    "confidence": 0.0,
                    "sub_questions": sub_questions,
                    "conflicting_info": []
                }

            # Step 4: Score relevance
            scored = await self._score_relevance(query, deduped)
            scored.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
            top_sources = scored[:max_sources]
            
            # Step 5: Synthesize
            report = await self._synthesize(query, sub_questions, top_sources)
            return report
        except Exception as e:
            logger.exception("ResearchAgent failed.")
            return {
                "summary": f"Research failed due to an error: {e}",
                "key_findings": [],
                "sources": [],
                "confidence": 0.0,
                "sub_questions": getattr(self, "_last_sub_questions", [query]),
                "conflicting_info": []
            }

    async def _decompose_query(self, query: str, max_sub_questions: int) -> List[str]:
        prompt = (
            f"Break this research query into {max_sub_questions} specific, independent, searchable sub-questions.\n"
            "JSON only: {\"sub_questions\": [\"q1\", \"q2\", ...]}\n"
            f"Query: {query}"
        )
        try:
            from langchain_core.messages import HumanMessage
            from models.router import Tier
            response, _ = await self.router.call_with_rotation(tier=Tier.REASONING, messages=[HumanMessage(content=prompt)])
            
            import re
            text = response.content
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(text)
                
            questions = data.get("sub_questions", [])
            if not questions:
                return [query]
            return questions[:max_sub_questions]
        except Exception as e:
            logger.warning(f"Decomposition failed, using original query: {e}")
            return [query]

    async def _score_relevance(self, query: str, results: List[dict]) -> List[dict]:
        if not results:
            return []
            
        sources_text = "\n".join(
            f"[{i}] {r.get('title', '')} - {r.get('snippet', '')}" 
            for i, r in enumerate(results)
        )
        prompt = (
            f"Score each result's relevance to the query '{query}' (0.0-1.0).\n"
            "JSON only: {\"scores\": [0.8, 0.3, ...]}\n\n"
            f"Results:\n{sources_text}"
        )
        try:
            from langchain_core.messages import HumanMessage
            from models.router import Tier
            response, _ = await self.router.call_with_rotation(tier=Tier.SPEED, messages=[HumanMessage(content=prompt)], temperature=0.0)
            
            import re
            text = response.content
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(text)
                
            scores = data.get("scores", [])
            # Map back
            for i, r in enumerate(results):
                r["relevance_score"] = float(scores[i]) if i < len(scores) else 0.5
        except Exception as e:
            logger.warning(f"Relevance scoring failed, defaulting to 0.5: {e}")
            for r in results:
                r["relevance_score"] = 0.5
                
        return results

    async def _synthesize(self, query: str, sub_questions: List[str], sources: List[dict]) -> dict:
        sources_text = "\n\n".join(
            f"Source [{i}]: {r.get('title', '')}\nURL: {r.get('url', '')}\nContent: {r.get('snippet', '')}" 
            for i, r in enumerate(sources)
        )
        prompt = (
            f"Create a structured research report for the query: '{query}'.\n"
            "Return JSON: {\"summary\": \"...\", \"key_findings\": [{\"claim\": \"...\", \"confidence\": 0.9, \"source_index\": 0}], \"conflicting_info\": [\"...\"], \"overall_confidence\": 0.8}\n\n"
            f"Based on these sources:\n{sources_text}"
        )
        try:
            from langchain_core.messages import HumanMessage
            from models.router import Tier
            response, _ = await self.router.call_with_rotation(tier=Tier.REASONING, messages=[HumanMessage(content=prompt)])
            
            import re
            text = response.content
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(text)
                
            # Build report
            source_list = [{"title": s.get("title", ""), "url": s.get("url", "")} for s in sources]
            
            return {
                "summary": data.get("summary", ""),
                "key_findings": data.get("key_findings", []),
                "sources": source_list,
                "confidence": float(data.get("overall_confidence", 0.5)),
                "sub_questions": sub_questions,
                "conflicting_info": data.get("conflicting_info", [])
            }
        except Exception as e:
            logger.warning(f"Synthesize failed, returning raw: {e}")
            source_list = [{"title": s.get("title", ""), "url": s.get("url", "")} for s in sources]
            return {
                "summary": "Failed to parse structured report.",
                "key_findings": [],
                "sources": source_list,
                "confidence": 0.0,
                "sub_questions": sub_questions,
                "conflicting_info": []
            }
