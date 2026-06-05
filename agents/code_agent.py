"""
agents.code_agent
=================
Generates, runs, and evaluates Python code in the sandbox.
Handles self-healing execution loops.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from config.config import get_config
from shared.constants import TaskType
from tools.code_interpreter import code_interpreter
from models.router import Tier

logger = logging.getLogger(__name__)


class CodeAgent:
    """Agent for code generation and self-healing execution."""

    def __init__(self, router: Any):
        self.router = router
        self.config = get_config()
        self.max_retries = getattr(self.config.code_agent, "max_execution_retries", 3)

    async def generate_and_run(self, task_description: str, context: str = None) -> Dict[str, Any]:
        last_error = ""
        code = ""
        
        for attempt in range(1, self.max_retries + 2):
            code = await self._generate_code(task_description, context, last_error, attempt)
            timeout = getattr(self.config.tools, "code_execution_timeout_sec", 15)
            
            exec_result = await code_interpreter(code, timeout_sec=timeout)
            
            if exec_result.get("success", False):
                explanation = await self._explain_code(code, exec_result.get("stdout", ""))
                return {
                    **exec_result,
                    "explanation": explanation,
                    "attempts": attempt
                }
                
            last_error = exec_result.get("stderr", "Unknown error")
            if attempt > self.max_retries:
                break
                
        explanation = await self._explain_failure(code, last_error)
        return {
            "code": code, 
            "stdout": "", 
            "stderr": last_error,
            "success": False, 
            "execution_time_ms": 0,
            "explanation": explanation, 
            "attempts": self.max_retries + 1
        }

    async def _generate_code(self, task: str, context: str, last_error: str, attempt: int) -> str:
        prompt = f"Task: {task}\n\n"
        if context:
            prompt += f"Context:\n{context}\n\n"
            
        if last_error and attempt > 1:
            prompt += f"Previous attempt failed with error:\n{last_error}\n\nPlease fix the code.\n\n"
            
        prompt += "Return ONLY the python code. Do not include markdown code fences."
        
        try:
            from langchain_core.messages import HumanMessage
            response, _ = await self.router.call_with_rotation(tier=Tier.CODE, messages=[HumanMessage(content=prompt)])
            
            code = response.content.strip()
            if code.startswith("```"):
                lines = code.split("\n")
                if len(lines) >= 2 and lines[0].startswith("```"):
                    if lines[-1].startswith("```"):
                        code = "\n".join(lines[1:-1])
            return code
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return "print('Error generating code')"

    async def _explain_code(self, code: str, stdout: str) -> str:
        prompt = (
            f"Explain this code and its output in 2 sentences.\n"
            f"Code:\n{code}\n\n"
            f"Output:\n{stdout}"
        )
        try:
            from langchain_core.messages import HumanMessage
            response, _ = await self.router.call_with_rotation(tier=Tier.SPEED, messages=[HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception:
            return "Executed successfully."

    async def _explain_failure(self, code: str, error: str) -> str:
        prompt = (
            f"Explain what went wrong in 2 sentences and how it might be fixed.\n"
            f"Code:\n{code}\n\n"
            f"Error:\n{error}"
        )
        try:
            from langchain_core.messages import HumanMessage
            response, _ = await self.router.call_with_rotation(tier=Tier.SPEED, messages=[HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception:
            return "Failed to execute code."
