"""
core.executor
=============
Tool execution node for the ARIA agent graph.

Runs the execution plan produced by the Planner:
- Consecutive parallelizable steps run under asyncio.gather()
- Non-parallelizable steps run sequentially
- Every step is wrapped with a configurable timeout
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, List

from config.config import get_config
from core.state import ARIAState
from shared.types import ErrorEntry, ToolCallRecord, ToolOutputRecord

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _group_steps(plan) -> list[list]:
    """
    Split a plan into execution batches.

    Consecutive steps with can_parallelize=True are grouped into a single
    batch (run with asyncio.gather). Every sequential step is its own batch.
    """
    batches: list[list] = []
    current_batch: list = []

    for step in plan:
        if step.can_parallelize:
            current_batch.append(step)
        else:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
            batches.append([step])

    if current_batch:
        batches.append(current_batch)

    return batches


async def _run_step(step, tool_registry: Any, user_id: str) -> ToolOutputRecord:
    """Execute a single plan step and return a ToolOutputRecord."""
    config = get_config()
    timeout_sec = getattr(config.tools, "default_tool_timeout_sec", 30)
    start = asyncio.get_event_loop().time()

    # Inject user_id from active state if missing
    if "user_id" not in step.tool_params or not step.tool_params["user_id"]:
        step.tool_params["user_id"] = user_id

    try:
        result = await asyncio.wait_for(
            tool_registry.call(step.tool, step.tool_params),
            timeout=timeout_sec,
        )
        latency_ms = int((asyncio.get_event_loop().time() - start) * 1000)
        return ToolOutputRecord(
            tool_name=step.tool,
            input=step.tool_params,
            output=result,
            latency_ms=latency_ms,
            success=True,
        )
    except asyncio.TimeoutError:
        latency_ms = int(timeout_sec * 1000)
        logger.warning("Tool '%s' timed out after %ds", step.tool, timeout_sec)
        return ToolOutputRecord(
            tool_name=step.tool,
            input=step.tool_params,
            output=None,
            latency_ms=latency_ms,
            success=False,
            error_message=f"Timeout after {timeout_sec}s",
        )
    except Exception as exc:
        latency_ms = int((asyncio.get_event_loop().time() - start) * 1000)
        logger.warning("Tool '%s' raised %s: %s", step.tool, type(exc).__name__, exc)
        return ToolOutputRecord(
            tool_name=step.tool,
            input=step.tool_params,
            output=None,
            latency_ms=latency_ms,
            success=False,
            error_message=str(exc),
        )


async def execute_plan(state: ARIAState, tool_registry: Any) -> ARIAState:
    """
    LangGraph node: execute every step in the execution plan.

    Parallelizable steps run concurrently. All others run sequentially.
    Failures are recorded but never crash the graph.

    Args:
        state:         Current ARIAState.
        tool_registry: ToolRegistry instance.

    Returns:
        Updated ARIAState with tool_outputs populated.
    """
    plan: list = state.get("execution_plan") or []
    tool_outputs: List[ToolOutputRecord] = list(state.get("tool_outputs") or [])
    tool_calls: List[ToolCallRecord] = list(state.get("tool_calls") or [])
    error_log: List[ErrorEntry] = list(state.get("error_log") or [])
    debug_trace: list[str] = list(state.get("debug_trace") or [])

    if not plan:
        debug_trace.append("Executor: no plan to execute")
        return {**state, "tool_outputs": tool_outputs, "debug_trace": debug_trace}

    batches = _group_steps(plan)

    for batch in batches:
        # Record call intent before executing
        for step in batch:
            tool_calls.append(
                ToolCallRecord(
                    tool_name=step.tool,
                    input=step.tool_params,
                    timestamp=_now_iso(),
                )
            )

        user_id = state.get("user_id", "")
        if len(batch) == 1:
            records = [await _run_step(batch[0], tool_registry, user_id)]
        else:
            records = list(
                await asyncio.gather(*[_run_step(s, tool_registry, user_id) for s in batch])
            )

        for record in records:
            tool_outputs.append(record)
            if not record.success:
                error_log.append(
                    ErrorEntry(
                        timestamp=_now_iso(),
                        component=f"Executor/{record.tool_name}",
                        error=record.error_message or "Unknown error",
                    )
                )

    success_count = sum(1 for r in tool_outputs if r.success)
    debug_trace.append(
        f"Executor: {len(tool_outputs)} tool call(s), "
        f"{success_count} succeeded, "
        f"{len(tool_outputs) - success_count} failed"
    )

    return {
        **state,
        "tool_calls": tool_calls,
        "tool_outputs": tool_outputs,
        "error_log": error_log,
        "debug_trace": debug_trace,
    }
