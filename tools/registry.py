"""
tools.registry
==============
Central registry for all ARIA tools.
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel

from config.config import get_config
from tools.calculator import calculator
from tools.code_interpreter import code_interpreter
from tools.deep_research import deep_research
from tools.document_rag import document_rag
from tools.memory_tool import memory_tool
from tools.structured_output import structured_output
from tools.summarizer import summarizer
from tools.web_search import web_search

logger = logging.getLogger(__name__)

class ToolSpec(BaseModel):
    """Specification for a registered tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    is_enabled: bool = True
    timeout_sec: int = 15

class ToolRegistry:
    """Central registry and executor for ARIA tools."""

    def __init__(self, config=None, router=None):
        self.config = config or get_config()
        self.router = router
        self._tools: Dict[str, Callable] = {}
        self._specs: Dict[str, ToolSpec] = {}
        self._register_all()

    def inject(self, **dependencies) -> None:
        """
        Update registered tools with runtime dependencies.
        Called after all components are initialized to wire in agents.
        """
        if "rag_agent" in dependencies:
            from tools.document_rag import document_rag
            self._tools["document_rag"] = lambda **kw: document_rag(
                rag_agent=dependencies["rag_agent"], **kw
            )

        if "memory_manager" in dependencies:
            from tools.memory_tool import memory_tool
            self._tools["memory_tool"] = lambda **kw: memory_tool(
                memory_manager=dependencies["memory_manager"], **kw
            )

    def _register_all(self):
        """Register all built-in tools based on configuration."""
        
        # 1. web_search
        self.register(
            "web_search",
            web_search,
            ToolSpec(
                name="web_search",
                description="Search the web for current information.",
                input_schema={"query": "str", "max_results": "int"},
                is_enabled=getattr(self.config.tools, "web_search_enabled", True)
            )
        )
        
        # 2. deep_research
        self.register(
            "deep_research",
            lambda **kwargs: deep_research(**kwargs, router=self.router),
            ToolSpec(
                name="deep_research",
                description="Conduct deep, multi-step research on a complex topic.",
                input_schema={"query": "str"},
                is_enabled=getattr(self.config.tools, "deep_research_enabled", True),
                timeout_sec=60
            )
        )
        
        # 3. document_rag
        self.register(
            "document_rag",
            lambda **kwargs: {"error": "document_rag tool not injected"},
            ToolSpec(
                name="document_rag",
                description="Search through uploaded user documents.",
                input_schema={"user_id": "str", "query": "str"},
                is_enabled=getattr(self.config.tools, "document_rag_enabled", True)
            )
        )
        
        # 4. code_interpreter
        self.register(
            "code_interpreter",
            code_interpreter,
            ToolSpec(
                name="code_interpreter",
                description="Execute Python code in a secure sandbox.",
                input_schema={"code": "str"},
                is_enabled=getattr(self.config.tools, "code_interpreter_enabled", True),
                timeout_sec=15
            )
        )
        
        # 5. memory_tool
        self.register(
            "memory_tool",
            lambda **kwargs: {"error": "memory_tool not injected"},
            ToolSpec(
                name="memory_tool",
                description="Read, write, or manage user's episodic memory.",
                input_schema={"operation": "str", "user_id": "str", "content": "str"},
                is_enabled=getattr(self.config.tools, "memory_tool_enabled", True)
            )
        )
        
        # 6. summarizer
        self.register(
            "summarizer",
            lambda **kwargs: summarizer(**kwargs, router=self.router),
            ToolSpec(
                name="summarizer",
                description="Summarize large blocks of text or URLs.",
                input_schema={"content": "str", "url": "str"},
                is_enabled=getattr(self.config.tools, "summarizer_enabled", True),
                timeout_sec=30
            )
        )
        
        # 7. structured_output
        self.register(
            "structured_output",
            lambda **kwargs: structured_output(**kwargs, router=self.router),
            ToolSpec(
                name="structured_output",
                description="Format text into structured formats like json, table, or bullets.",
                input_schema={"content": "str", "format_type": "str"},
                is_enabled=getattr(self.config.tools, "structured_output_enabled", True)
            )
        )
        
        # 8. calculator
        self.register(
            "calculator",
            calculator,
            ToolSpec(
                name="calculator",
                description="Perform exact mathematical calculations safely.",
                input_schema={"expression": "str"},
                is_enabled=getattr(self.config.tools, "calculator_enabled", True),
                timeout_sec=5
            )
        )

    def register(self, name: str, fn: Callable, spec: ToolSpec):
        """Register a new tool."""
        self._tools[name] = fn
        self._specs[name] = spec

    async def call(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Call a registered tool safely with logging and timeouts."""
        if tool_name not in self._tools:
            available = ", ".join(self._tools.keys())
            raise KeyError(f"Tool '{tool_name}' not found. Available tools: {available}")
            
        spec = self._specs[tool_name]
        if not spec.is_enabled:
            raise RuntimeError(f"Tool '{tool_name}' is disabled in config")

        fn = self._tools[tool_name]
        start_time = time.monotonic()
        success = True
        
        try:
            if asyncio.iscoroutinefunction(fn) or (
                hasattr(fn, "__name__") and fn.__name__ == "<lambda>" and "async" in str(fn)
                # Lambdas wrapping async functions might need special handling
            ) or (
                tool_name in ["web_search", "deep_research", "document_rag", 
                              "code_interpreter", "memory_tool", "summarizer", 
                              "structured_output"]
            ):
                result = await asyncio.wait_for(fn(**params), timeout=spec.timeout_sec)
            else:
                # Sync function execution
                loop = asyncio.get_running_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(executor, lambda: fn(**params)),
                        timeout=spec.timeout_sec
                    )
            return result
        except asyncio.TimeoutError:
            success = False
            return {"error": f"Tool timed out after {spec.timeout_sec}s"}
        except Exception as e:
            success = False
            logger.error(f"Tool {tool_name} failed: {e}")
            return {"error": f"{type(e).__name__}: {str(e)}"}
        finally:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                f"ToolCall: {tool_name} | keys: {list(params.keys())} | "
                f"latency: {latency_ms:.1f}ms | success: {success}"
            )

    @classmethod
    def list_tools(cls) -> str:
        """List all available tools and their descriptions."""
        registry = cls()  # Instantiate to load default specs
        lines = []
        for name, spec in registry._specs.items():
            if spec.is_enabled:
                lines.append(f"- {name}: {spec.description}")
        return "\n".join(lines)
